# bootstrap_neo4j.py
"""
Minimal importer: loads output/output.json into Neo4j.
Accepts relationship endpoints in either shape:
  - {"label":"Manager","properties":{"name":"Alice"}}
  - {"label":"Manager","name":"Alice"}   # legacy
"""

import json
import os
from typing import Any, Dict, Tuple
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "lolkola93")
GRAPH_PATH = "output/output.json"

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prune_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

def connect_driver():
    if NEO4J_PASSWORD or NEO4J_USER:
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return GraphDatabase.driver(NEO4J_URI)

def ensure_basic_constraints(driver):
    stmts = [
        "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT manager_name IF NOT EXISTS FOR (m:Manager) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT engineer_name IF NOT EXISTS FOR (e:Engineer) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT part_pid IF NOT EXISTS FOR (p:Part) REQUIRE p.part_id IS UNIQUE"
    ]
    with driver.session() as s:
        for cy in stmts:
            s.run(cy)

def merge_node(session, label: str, props: Dict[str, Any]):
    props = prune_none(props)
    if label == "Part" and "part_id" in props:
        session.run(
            f"MERGE (n:`{label}` {{part_id:$id}}) SET n += $props",
            id=props["part_id"], props=props
        ); return
    if "name" in props:
        session.run(
            f"MERGE (n:`{label}` {{name:$id}}) SET n += $props",
            id=props["name"], props=props
        ); return

    # Literal map MERGE (can't MERGE (n:Label $props))
    if not props:
        raise ValueError(f"Cannot MERGE `{label}` without an identity or any properties.")
    keys = sorted(props.keys())
    map_lit = ", ".join([f"{k}: ${'p_' + k}" for k in keys])
    param_map = {("p_" + k): props[k] for k in keys}
    param_map["props"] = props
    cypher = f"MERGE (n:`{label}` {{ {map_lit} }}) SET n += $props"
    session.run(cypher, **param_map)

def normalize_endpoint(n: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Accept either:
      {"label":"X","properties":{...}}
      {"label":"X","name":"..."} / {"label":"X","part_id":"..."}  (legacy)
    """
    label = n.get("label")
    props = {}
    if isinstance(n.get("properties"), dict):
        props.update(n["properties"])
    # legacy top-level identity fallbacks
    for k in ("name", "part_id"):
        if k in n and n[k] is not None:
            props[k] = n[k]
    return label, prune_none(props)

def merge_relationship(session, r: Dict[str, Any]):
    f_label, f_props = normalize_endpoint(r["from"])
    t_label, t_props = normalize_endpoint(r["to"])
    rtype = r["type"]
    rprops = prune_none(r.get("properties", {}) or {})

    # Ensure endpoint nodes exist first
    merge_node(session, f_label, f_props)
    merge_node(session, t_label, t_props)

    # Build MATCH by identity
    if f_label == "Part" and "part_id" in f_props:
        f_match = f"(a:`{f_label}` {{part_id:$fa}})"; fa = f_props["part_id"]; f_where = None
    elif "name" in f_props:
        f_match = f"(a:`{f_label}` {{name:$fa}})"; fa = f_props["name"]; f_where = None
    else:
        f_match = f"(a:`{f_label}`)"; fa = None; f_where = "all(k IN keys($fprops) WHERE a[k] = $fprops[k])"

    if t_label == "Part" and "part_id" in t_props:
        t_match = f"(b:`{t_label}` {{part_id:$tb}})"; tb = t_props["part_id"]; t_where = None
    elif "name" in t_props:
        t_match = f"(b:`{t_label}` {{name:$tb}})"; tb = t_props["name"]; t_where = None
    else:
        t_match = f"(b:`{t_label}`)"; tb = None; t_where = "all(k IN keys($tprops) WHERE b[k] = $tprops[k])"

    where_clauses = []
    params: Dict[str, Any] = {"rprops": rprops}
    if fa is not None:
        params["fa"] = fa
    else:
        params["fprops"] = f_props
        where_clauses.append(f_where)
    if tb is not None:
        params["tb"] = tb
    else:
        params["tprops"] = t_props
        where_clauses.append(t_where)

    cypher = f"MATCH {f_match}, {t_match} "
    if where_clauses:
        cypher += "WHERE " + " AND ".join(where_clauses) + " "
    cypher += f"MERGE (a)-[r:`{rtype}`]->(b) SET r += $rprops"

    session.run(cypher, **params)

def import_output_json(graph_path: str):
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"File not found: {graph_path}")

    graph = read_json(graph_path)
    nodes = graph.get("nodes", [])
    rels  = graph.get("relationships", [])

    print(f"[IMPORT] Loading {len(nodes)} nodes and {len(rels)} relationships from {graph_path}")
    driver = connect_driver()
    ensure_basic_constraints(driver)

    with driver.session() as session:
        print("[IMPORT] Merging nodes…")
        for n in nodes:
            merge_node(session, n["label"], n.get("properties", {}) or {})

        print("[IMPORT] Merging relationships…")
        for r in rels:
            merge_relationship(session, r)

    driver.close()
    print("[IMPORT] Done.")

if __name__ == "__main__":
    import_output_json(GRAPH_PATH)
