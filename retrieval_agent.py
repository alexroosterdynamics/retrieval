# retrieval_agent.py
"""
Usage:
  python retrieval_agent.py

You will be prompted for a query that must start with either:
  - READ ...
  - MODIFY ...

The agent will:
  1) TOOL: extract nodes from the query (schema-aware).
  2) TOOL: for each node, pull top-3 similar items from Qdrant (facts).
  3) TOOL: combine facts + schema into an LLM plan. If READ -> answer only.
     If MODIFY -> update local output/output.json and also write to Neo4j.
"""

import os
from tools import (
    extract_nodes_from_query,
    vector_lookup_for_nodes,
    combine_facts_and_update,
)

def main():
    print("=== Retrieval Agent ===")
    print("Enter a query starting with READ or MODIFY.")
    query = input("Query> ").strip()

    if not (query.upper().startswith("READ") or query.upper().startswith("MODIFY")):
        print("Error: Query must start with READ or MODIFY.")
        return

    print("\n[STEP 1] Extracting nodes from query using TOOL 1 …")
    nodes = extract_nodes_from_query(query)
    print(f"[STEP 1] Nodes detected ({len(nodes)}): {nodes}")

    print("\n[STEP 2] Retrieving facts (top-3 per node) from vector store using TOOL 2 …")
    facts = vector_lookup_for_nodes(nodes, top_k=3)
    for k, v in facts.items():
        print(f"[STEP 2] FACTS for '{k}': {len(v)} hits")

    print("\n[STEP 3] Combining facts + schema and applying plan using TOOL 3 …")
    result = combine_facts_and_update(
        query=query,
        facts=facts,
        schema_path="schema.json",
        output_path="output/output.json"
    )

    print("\n=== RESULT ===")
    print(f"Mode: {result.get('mode')}")
    print("Answer:")
    print(result.get("answer", ""))
    print("\nGraph changes applied (local backup & Neo4j if MODIFY):")
    print(result.get("graph_changes", {}))

if __name__ == "__main__":
    main()
