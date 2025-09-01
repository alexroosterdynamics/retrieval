# tools.py
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from neo4j import GraphDatabase

# -------- Config & Clients --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "emails_past")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lolkola93")

oaiclient = OpenAI(api_key=OPENAI_API_KEY)
qclient = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD) if NEO4J_PASSWORD or NEO4J_USER else None)

# -------- File helpers --------
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------- Schema helpers --------
def load_schema(path: str = "schema.json") -> Dict[str, Any]:
    return read_json(path)

def _schema_maps(schema: Dict[str, Any]):
    node_defs = {n["label"]: n.get("properties", {}) for n in schema.get("nodes", [])}
    rel_defs = {r["type"]: r.get("properties", {}) for r in schema.get("relationships", [])}
    return node_defs, rel_defs

# -------- OpenAI helpers --------
def embed_text(text: str) -> List[float]:
    emb = oaiclient.embeddings.create(model=EMBED_MODEL, input=text)
    return emb.data[0].embedding

def gpt_json(system: str, user: str) -> Dict[str, Any]:
    resp = oaiclient.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception:
        return {}

# =========================================================
# TOOL 1: Find all node mentions from user query (by schema)
# =========================================================
def extract_nodes_from_query(query: str, schema_path: str = "schema.json") -> List[Dict[str, Any]]:
    """
    Returns a list of node candidates like:
      [{"label": "Project", "properties": {"name": "Apollo"}},
       {"label": "Engineer", "properties": {"name": "Bob"}}]
    Only labels/props allowed by schema. Values must appear in the query.
    """
    schema = load_schema(schema_path)
    system = (
        "You identify node mentions for a graph. Only use labels/props from the SCHEMA. "
        "Return JSON: {\"nodes\":[{\"label\":\"...\",\"properties\":{...}}, ...]}. "
        "Include only properties and values explicitly found in the query."
    )
    user = f"SCHEMA:\n{json.dumps(schema)}\n\nQUERY:\n{query}"
    result = gpt_json(system, user)
    nodes = result.get("nodes", [])
    # prune to schema
    node_defs, _ = _schema_maps(schema)
    out = []
    for n in nodes:
        label = n.get("label")
        if label not in node_defs:
            continue
        props = n.get("properties", {}) or {}
        allowed = node_defs[label]
        pruned = {k: v for k, v in props.items() if k in allowed}
        out.append({"label": label, "properties": pruned})
    return out

# =================================================================
# TOOL 2: For each node mention, query vector store and get top 3
# =================================================================
def vector_lookup_for_nodes(nodes: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each node dict, build a short query string (e.g., 'Project Apollo'),
    embed it, search Qdrant, return a dict:
      key = canonical string for node
      value = list of top-k {score, payload, vector}
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    for node in nodes:
        label = node.get("label")
        props = node.get("properties", {})
        # Canonical key for display
        disp = f'{label} ' + " ".join(f'{k}:{v}' for k, v in props.items())
        query_text = disp.strip()
        print(f"[TOOL2] Querying vector store for: {query_text}")
        qvec = embed_text(query_text)

        # Qdrant search
        hits = qclient.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=qvec,
            limit=top_k,
            with_payload=True,
            with_vectors=True
        )
        results[disp] = []
        for h in hits:
            results[disp].append({
                "score": h.score,
                "payload": h.payload,
                "vector": h.vector
            })
        print(f"[TOOL2] Found {len(results[disp])} results for {disp}")
    return results

# =========================================================================================
# TOOL 3: Combine facts into prompt; update /output/output.json; execute Neo4j if MODIFY
# =========================================================================================
def combine_facts_and_update(
    query: str,
    facts: Dict[str, List[Dict[str, Any]]],
    schema_path: str = "schema.json",
    output_path: str = "output/output.json"
) -> Dict[str, Any]:
    """
    Build a structured prompt that treats 'facts' as ground truth (top-3 per node),
    asks GPT to output either a READ answer or a MODIFY plan with graph changes,
    then (a) update local output.json, and (b) run Neo4j writes if MODIFY.
    """
    schema = load_schema(schema_path)

    # Gather compact facts (just the triples + key fields) to reduce token noise
    compact_facts = {}
    for key, lst in facts.items():
        compact = []
        for item in lst:
            payload = item.get("payload", {})
            compact.append({
                "email_id": payload.get("email_id"),
                "subject": payload.get("subject"),
                "triples": payload.get("triples", []),
                "score": item.get("score")
            })
        compact_facts[key] = compact

    system = (
        "You are a graph update agent. You receive a SCHEMA, a USER_QUERY (starts with READ or MODIFY), "
        "and FACTS (ground truth from a vector store: triples & context). "
        "Respect the SCHEMA strictly: only allowed labels, relationships, and properties. "
        "If READ: produce an 'answer' summarizing facts; do NOT modify graph. "
        "If MODIFY: produce a set of graph changes (MERGE semantics) to apply. "
        "Return strictly this JSON shape:\n"
        "{\n"
        '  "mode": "READ" | "MODIFY",\n'
        '  "answer": "string",\n'
        '  "graph_changes": {\n'
        '    "nodes_to_merge": [ {"label":"...","properties":{...}}, ... ],\n'
        '    "relationships_to_merge": [\n'
        '       {"from":{"label":"...","properties":{...}}, "type":"...", "to":{"label":"...","properties":{...}}, "properties":{...}}\n'
        "    ]\n"
        "  }\n"
        "}\n"
        "Include only properties that appear in the USER_QUERY or FACTS."
    )
    user = f"SCHEMA:\n{json.dumps(schema)}\n\nUSER_QUERY:\n{query}\n\nFACTS:\n{json.dumps(compact_facts)}"

    print("[TOOL3] Asking GPT to plan READ/MODIFY based on facts + schema…")
    plan = gpt_json(system, user)

    # Validate and prune to schema
    node_defs, rel_defs = _schema_maps(schema)
    def prune_node(n: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(n, dict):
            return None
        label = n.get("label")
        if label not in node_defs:
            return None
        props = n.get("properties", {}) or {}
        allowed = node_defs[label]
        return {"label": label, "properties": {k: v for k, v in props.items() if k in allowed}}

    pruned_nodes = []
    for n in (plan.get("graph_changes", {}).get("nodes_to_merge", []) or []):
        pn = prune_node(n)
        if pn:
            pruned_nodes.append(pn)

    pruned_rels = []
    for r in (plan.get("graph_changes", {}).get("relationships_to_merge", []) or []):
        rtype = r.get("type")
        if rtype not in rel_defs:
            continue
        f = prune_node(r.get("from", {}))
        t = prune_node(r.get("to", {}))
        if not f or not t:
            continue
        allowed_rprops = rel_defs[rtype]
        rprops = r.get("properties", {}) or {}
        pruned_rels.append({
            "from": f,
            "type": rtype,
            "to": t,
            "properties": {k: v for k, v in rprops.items() if k in allowed_rprops}
        })

    mode = str(plan.get("mode", "READ")).upper()
    answer = plan.get("answer", "")

    # Update local output.json (backup graph)
    print(f"[TOOL3] Loading local graph backup: {output_path}")
    try:
        graph = read_json(output_path)
    except Exception:
        graph = {"nodes": [], "relationships": []}

    print("[TOOL3] Merging nodes into local graph JSON…")
    graph = merge_nodes(graph, pruned_nodes)

    print("[TOOL3] Merging relationships into local graph JSON…")
    graph = merge_relationships(graph, pruned_rels)

    write_json(output_path, graph)
    print(f"[TOOL3] Local backup updated: {output_path}")

    # Neo4j execution
    if mode == "MODIFY":
        print("[TOOL3] Executing MODIFY in Neo4j…")
        neo4j_merge(graph_changes={"nodes_to_merge": pruned_nodes, "relationships_to_merge": pruned_rels})
    else:
        print("[TOOL3] READ mode — no Neo4j writes.")

    return {"mode": mode, "answer": answer, "graph_changes": {"nodes_to_merge": pruned_nodes, "relationships_to_merge": pruned_rels}}

# -------- Local graph merge helpers --------
def _node_key(n: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    label = n["label"]
    props = n.get("properties", {})
    # Identity rule: Part->part_id, otherwise -> name if present, else full prop set
    if label == "Part" and "part_id" in props:
        ident = (("part_id", props["part_id"]),)
    elif "name" in props:
        ident = (("name", props["name"]),)
    else:
        ident = tuple(sorted(props.items()))
    return (label, ident)

def merge_nodes(graph: Dict[str, Any], nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    existing = graph.get("nodes", [])
    by_key = {_node_key(n): n for n in existing}
    for n in nodes:
        k = _node_key(n)
        if k in by_key:
            # shallow merge of properties
            by_key[k]["properties"].update(n.get("properties", {}))
        else:
            existing.append(n)
            by_key[k] = n
    graph["nodes"] = existing
    return graph

def _rel_key(r: Dict[str, Any]) -> Tuple:
    return (
        _node_key(r["from"]),
        r["type"],
        _node_key(r["to"]),
        # We treat relationships as versioned by validity properties if present
        tuple(sorted((r.get("properties") or {}).items()))
    )

def merge_relationships(graph: Dict[str, Any], rels: List[Dict[str, Any]]) -> Dict[str, Any]:
    existing = graph.get("relationships", [])
    existing_keys = {_rel_key(r): i for i, r in enumerate(existing)}
    for r in rels:
        k = _rel_key(r)
        if k in existing_keys:
            # merge props shallowly
            idx = existing_keys[k]
            existing[idx]["properties"].update(r.get("properties", {}))
        else:
            existing.append(r)
            existing_keys[k] = len(existing) - 1
    graph["relationships"] = existing
    return graph

# -------- Neo4j writers --------
def _neo4j_merge_node(tx, label: str, props: Dict[str, Any]):
    # identity same as local merge helper
    if label == "Part" and "part_id" in props:
        ident_key = "part_id"
    elif "name" in props:
        ident_key = "name"
    else:
        # fallback: use all props as identity (not ideal, but consistent with local)
        ident_key = None

    if ident_key:
        cypher = f"MERGE (n:`{label}` {{ {ident_key}: $ident }}) SET n += $props RETURN n"
        tx.run(cypher, ident=props[ident_key], props=props)
    else:
        # MERGE on full props map
        cypher = f"MERGE (n:`{label}` $props) RETURN n"
        tx.run(cypher, props=props)

def _neo4j_merge_relationship(tx, r: Dict[str, Any]):
    f = r["from"]; t = r["to"]; rtype = r["type"]; rprops = r.get("properties", {})
    f_label = f["label"]; f_props = f.get("properties", {})
    t_label = t["label"]; t_props = t.get("properties", {})

    def ident_clause(label: str, props: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if label == "Part" and "part_id" in props:
            return f"`{label}` {{part_id: $pid}}", {"pid": props["part_id"]}
        if "name" in props:
            return f"`{label}` {{name: $name}}", {"name": props["name"]}
        # fallback: match on full map using ALL k IN keys() …
        return f"`{label}`", {"full": props}

    fpat, fbind = ident_clause(f_label, f_props)
    tpat, tbind = ident_clause(t_label, t_props)

    # Build MERGE and SET
    cypher = (
        f"MATCH (a:{fpat}) MATCH (b:{tpat}) "
        f"MERGE (a)-[r:`{rtype}`]->(b) "
        f"SET r += $rprops "
        f"RETURN type(r) AS type"
    )
    params = {}
    params.update({k if k != "full" else "fprops": v for k, v in fbind.items()})
    params.update({("t" + k) if k != "full" else "tprops": v for k, v in tbind.items()})
    params["rprops"] = rprops

    # replace placeholders if we used full map
    cypher = cypher.replace("`{label}`", "").replace("{part_id: $pid}", "").replace("{name: $name}", "")
    # For simplicity, if we fell back to full map, do MERGE nodes beforehand
    # (we already MERGE nodes before calling this)

    tx.run(cypher, **params)

def neo4j_merge(graph_changes: Dict[str, Any]):
    nodes = graph_changes.get("nodes_to_merge", [])
    rels = graph_changes.get("relationships_to_merge", [])

    with neo4j_driver.session() as session:
        def work(tx):
            for n in nodes:
                _neo4j_merge_node(tx, n["label"], n.get("properties", {}))
            for r in rels:
                # ensure nodes exist
                _neo4j_merge_node(tx, r["from"]["label"], r["from"].get("properties", {}))
                _neo4j_merge_node(tx, r["to"]["label"], r["to"].get("properties", {}))
                _neo4j_merge_relationship(tx, r)
        session.execute_write(work)

# -------- Convenience: simple read against Neo4j --------
def neo4j_read_example(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    with neo4j_driver.session() as session:
        result = session.run(cypher, **(params or {}))
        return [r.data() for r in result]
