# GraphITI — End-to-End Mini Docs

Tiny system to turn unstructured emails into a **Neo4j project graph** + a **Qdrant vector index**, with a CLI agent to **READ** or **MODIFY**.

---

## 🚀 How it fits together

raw_data/past_emails.json → make_embedings.py → Qdrant (emails_past)
↘ output/output.json → Neo4j
retrieval_agent.py (extract → retrieve → plan) → uses both Qdrant + Neo4j

yaml
Copy code

- **`schema.json`** — defines allowed nodes/relationships
- **`tools.py`** — implements Tools 1/2/3 (extract, retrieve, plan)
- **`bootstrap_neo4j.py`** — seeds Neo4j from local backup
- **Qdrant** — vector database (Docker)
- **Neo4j** — source of truth (Desktop or Server)

---

## ⚡ Quick Start

### 1. Start services

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
# Neo4j running at bolt://localhost:7687
2. Set environment
powershell
Copy code
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:NEO4J_PASSWORD="yourPassword"
3. Embed & index historical emails
bash
Copy code
python make_embedings.py   # creates Qdrant collection with vectors + triples
4. Seed Neo4j from local backup (optional)
bash
Copy code
python bootstrap_neo4j.py  # imports output/output.json
5. Ask the agent
bash
Copy code
python retrieval_agent.py

Query> READ what parts does Project Apollo use?

Query> MODIFY set Apollo status to on_hold from 2025-09-05
📖 What happens on READ vs MODIFY
READ
Extracts node mentions (schema-aware).

Pulls top-3 matching facts per node from Qdrant.

Builds an answer using facts as ground truth.

❌ No writes to Neo4j, ❌ no change to output/output.json.

MODIFY
Extracts nodes + retrieves facts.

Produces a schema-validated plan (nodes_to_merge, relationships_to_merge).

Updates output/output.json (local backup).

Executes corresponding MERGE writes in Neo4j.

🛡 Validation & Safety Rails
The agent validates all AI outputs against your schema.json:

Node extraction (TOOL 1) → drops unknown labels & disallowed props.

Plan application (TOOL 3) → prunes relationships/nodes not in schema.

Neo4j merges use identity rules:

Part.part_id

or name (for others)

fallbacks use literal maps (never MERGE (n:Label $props)).

📂 Files Overview
File	Purpose
schema.json	Allowed node labels, relationship types, and properties
raw_data/past_emails.json	30 condensed historical emails (mid-project state)
make_embedings.py	Concatenate email text, extract triples via GPT, embed, upsert into Qdrant
output/output.json	Local JSON backup of the current graph
bootstrap_neo4j.py	One-shot importer of output.json into Neo4j
tools.py	Implements Tools 1/2/3, vector search, schema validation, merges
retrieval_agent.py	CLI agent orchestrating extract → retrieve → plan (READ/MODIFY)

💡 Tips
If Neo4j errors on MERGE → ensure nodes have identities (name or part_id).

Confirm Qdrant is running at http://localhost:6333 and QDRANT_COLLECTION matches.

Use ISO dates (YYYY-MM-DD) for relationship validity (valid_from / valid_to).

🔑 Required Environment Variables
OPENAI_API_KEY

NEO4J_URI (default: bolt://localhost:7687)

NEO4J_USER (default: neo4j)

NEO4J_PASSWORD

QDRANT_HOST (default: localhost)

QDRANT_PORT (default: 6333)

QDRANT_COLLECTION (default: emails_past)

EMBED_MODEL (default: text-embedding-3-small)

LLM_MODEL (default: gpt-4o-mini)

✨ Built for clarity. Keep your schema tight, and your writes boring.

yaml
Copy code

---

👉 Save this as `README.md` in your project root, commit, and push.
GitHub will automatically render it on your repository’s homepage.

Want me to also include a **`.gitignore`** (Python + VS Code + Neo4j/Qdrant artifacts) so your repo stays clean?
```
