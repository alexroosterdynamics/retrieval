# make_embedings.py
"""
Create a Qdrant collection and embed emails from raw_data/past_emails.json.
For each email, we:
  1) Build a natural-language text (From/To/CC/BCC/Subject/Body).
  2) Ask GPT to extract graph triples (node --relationship--> node) using schema.json
     and ONLY the allowed node & relationship types + properties from the schema.
  3) Generate an embedding for the text and upsert to Qdrant with the extracted triples as metadata.

Prereqs:
  pip install qdrant-client openai python-dateutil

Environment variables:
  OPENAI_API_KEY   (required)
  QDRANT_HOST      (default: localhost)
  QDRANT_PORT      (default: 6333)
  QDRANT_COLLECTION (default: "emails_past")
  EMBED_MODEL      (default: "text-embedding-3-small")
  LLM_MODEL        (default: "gpt-4o-mini")

Input files (expected to exist):
  - schema.json
  - raw_data/past_emails.json

Notes:
  - We derive the vector size from the first embedding and create the collection accordingly.
  - IDs in Qdrant are numeric; we map email IDs like "e-001" -> 1
  - We defensively FILTER GPT output to enforce the schema (allowed labels/relationships/properties only).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateparser
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# ---------------------------
# Configuration
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "emails_past")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# ---------------------------
# Helpers
# ---------------------------
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_numeric_id(email_id: str) -> int:
    """
    Map IDs like "e-001" -> 1. If no digits, hash fallback.
    """
    m = re.search(r"(\d+)$", email_id)
    if m:
        return int(m.group(1))
    # Fallback: simple deterministic hash
    return abs(hash(email_id)) % (2**31)


def build_email_text(email: Dict[str, Any]) -> str:
    subj = email.get("subject", "").strip()
    sender = email.get("from", "").strip()
    to_list = ", ".join(email.get("recipients", []) or [])
    cc_list = ", ".join(email.get("cc", []) or [])
    bcc_list = ", ".join(email.get("bcc", []) or [])
    when = email.get("send_date", "").strip()
    body = email.get("body", "").strip()

    parts = []
    parts.append(f"Email sent on {when}.")
    parts.append(f"From: {sender}.")
    parts.append(f"To: {to_list or 'none'}.")
    parts.append(f"CC: {cc_list or 'none'}.")
    parts.append(f"BCC: {bcc_list or 'none'}.")
    parts.append(f"Subject: {subj}.")
    parts.append("Body:")
    parts.append(body)
    return "\n".join(parts)


def gpt_extract_triples(client: OpenAI, schema: Dict[str, Any], email_text: str) -> Dict[str, Any]:
    """
    Ask GPT to extract triples that conform to the given schema.
    Output format (strict JSON):
    {
      "triples": [
        {
          "from": { "label": "...", "properties": { /* only allowed per schema & present in text */ } },
          "type": "WORKS_ON",
          "to":   { "label": "...", "properties": { /* as above */ } },
          "properties": { /* ONLY allowed relationship props & present in text */ }
        }
      ]
    }
    """
    system = (
        "You are an expert information extraction engine for a graph database. "
        "You will receive a SCHEMA and an EMAIL. Extract ONLY the triples (edges) that are "
        "explicitly stated or unambiguously implied by the email AND allowed by the SCHEMA. "
        "Do not invent nodes or relationships. Use only labels and relationship types listed in the schema. "
        "For node properties, include only keys defined for that node label and values that appear in the email. "
        "For relationship properties, include only keys defined for that relationship type and values that appear in the email. "
        "If a value is not present, omit that key. "
        "Return STRICT JSON with a single key 'triples' (array). No extra commentary."
    )

    user = f"""SCHEMA:
{json.dumps(schema, ensure_ascii=False)}

EMAIL:
{email_text}
"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "triples" in data and isinstance(data["triples"], list):
            return data
    except Exception:
        pass

    # Fallback: empty
    return {"triples": []}


def filter_triples_to_schema(
    triples: List[Dict[str, Any]],
    schema: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Enforce schema: keep only triples whose labels/types are allowed,
    and prune properties to allowed keys only.
    """
    # Derive allowed sets & property maps
    node_defs = {n["label"]: n.get("properties", {}) for n in schema.get("nodes", [])}
    rel_defs = {r["type"]: r.get("properties", {}) for r in schema.get("relationships", [])}

    allowed_node_labels = set(node_defs.keys())
    allowed_rel_types = set(rel_defs.keys())

    def prune_node(node_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(node_obj, dict):
            return None
        label = node_obj.get("label")
        if label not in allowed_node_labels:
            return None
        allowed_props = node_defs.get(label, {})
        props_in = node_obj.get("properties", {}) or {}
        # keep only allowed keys
        pruned = {k: v for k, v in props_in.items() if k in allowed_props}
        # Only keep node if it has at least one identifying property
        return {"label": label, "properties": pruned}

    filtered: List[Dict[str, Any]] = []
    for t in triples:
        rtype = t.get("type")
        if rtype not in allowed_rel_types:
            continue

        f = prune_node(t.get("from", {}))
        to = prune_node(t.get("to", {}))
        if not f or not to:
            continue

        # Relationship properties
        allowed_rprops = rel_defs.get(rtype, {})
        rprops_in = t.get("properties", {}) or {}
        rprops_out = {k: v for k, v in rprops_in.items() if k in allowed_rprops}

        filtered.append({
            "from": f,
            "type": rtype,
            "to": to,
            "properties": rprops_out
        })

    return filtered


def get_embedding(client: OpenAI, text: str) -> List[float]:
    emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return emb.data[0].embedding


def ensure_collection(qclient: QdrantClient, collection: str, vector_size: int):
    try:
        qclient.get_collection(collection)
        # If exists but incompatible size, drop & recreate
        # (Simplest path for demo; for production, consider migrate)
        info = qclient.get_collection(collection)
        # Heuristic check: if vectors config size differs, recreate.
        # qdrant-client returns different shapes depending on version; handle generically.
        have_size = None
        try:
            vcfg = info.config.params.get("vectors", {})
            if isinstance(vcfg, dict):
                have_size = vcfg.get("size")
        except Exception:
            pass

        if have_size is not None and have_size != vector_size:
            print(f"[i] Recreating collection '{collection}' with vector size {vector_size} (was {have_size})")
            qclient.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            # Assume OK if missing or matching
            pass
    except Exception:
        print(f"[i] Creating collection '{collection}' with vector size {vector_size}")
        qclient.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def main():
    schema = read_json("schema.json")
    emails = read_json("raw_data/past_emails.json")

    # Initialize clients
    oaiclient = OpenAI(api_key=OPENAI_API_KEY)
    qclient = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Prepare: compute first vector to determine dimension
    if not emails:
        print("No emails found in raw_data/past_emails.json")
        return

    first_text = build_email_text(emails[0])
    first_vec = get_embedding(oaiclient, first_text)
    dim = len(first_vec)
    ensure_collection(qclient, QDRANT_COLLECTION, dim)

    # Upsert the first email
    points = []
    triples_raw = gpt_extract_triples(oaiclient, schema, first_text)
    triples = filter_triples_to_schema(triples_raw.get("triples", []), schema)
    points.append({
        "id": to_numeric_id(emails[0]["id"]),
        "vector": first_vec,
        "payload": {
            "email_id": emails[0]["id"],
            "from": emails[0].get("from"),
            "subject": emails[0].get("subject"),
            "recipients": emails[0].get("recipients", []),
            "cc": emails[0].get("cc", []),
            "bcc": emails[0].get("bcc", []),
            "send_date": emails[0].get("send_date"),
            "text": first_text,
            "triples": triples
        }
    })

    # Process the rest
    for email in emails[1:]:
        text = build_email_text(email)
        vec = get_embedding(oaiclient, text)
        triples_raw = gpt_extract_triples(oaiclient, schema, text)
        triples = filter_triples_to_schema(triples_raw.get("triples", []), schema)

        points.append({
            "id": to_numeric_id(email["id"]),
            "vector": vec,
            "payload": {
                "email_id": email["id"],
                "from": email.get("from"),
                "subject": email.get("subject"),
                "recipients": email.get("recipients", []),
                "cc": email.get("cc", []),
                "bcc": email.get("bcc", []),
                "send_date": email.get("send_date"),
                "text": text,
                "triples": triples
            }
        })

        # Upsert in small batches to avoid huge request bodies
        if len(points) >= 16:
            qclient.upsert(collection_name=QDRANT_COLLECTION, points=points)
            print(f"[i] Upserted {len(points)} points")
            points = []

    # Flush any remaining
    if points:
        qclient.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"[i] Upserted {len(points)} points (final batch)")

    print(f"✅ Done. Collection '{QDRANT_COLLECTION}' populated with {len(emails)} embedded emails.")


if __name__ == "__main__":
    main()
