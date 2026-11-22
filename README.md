# FPTU RAG — Academic Assistant for FPT University

A chat assistant that answers questions about the FPT University AI curriculum — course syllabi, learning outcomes, assessments, schedules, prerequisites and specialisation tracks — in Vietnamese or English.

It combines **semantic vector search** (FAISS) with a **knowledge graph** of courses (Neo4j, or an in-memory graph as fallback), so it can answer multi-hop questions such as *"Tell me about SEG301 and its prerequisites"* or *"What is the path from CSI106 to AIG202c?"*.

## Architecture

```
                 Flask web app  (chat UI + REST API)
                              │
                     AdvancedRAGEngine
          ┌───────────────────┼────────────────────┐
     QueryRouter        Hybrid retrieval        QueryChain
  quick replies,     ┌────────┴─────────┐    multi-hop follow-up
  intent analysis    │                  │    queries, merged into
                FAISS vector       Knowledge graph     one answer
             (summary / chunk /   (Neo4j or in-memory)
              detail indexes)
                              │
                   Gemini 1.5 Flash answer generation
```

| Component | Role |
|---|---|
| `QueryPreprocessor` / `QueryRouter` | Normalises the question, answers greetings instantly, classifies intent (factual, listing, comparative, analytical, multi-hop) and picks a search strategy. |
| `HierarchicalIndex` | Three FAISS inner-product indexes — course summaries, section chunks and details — searched with per-type boost factors. |
| `GraphDatabase` | Course graph used for relationship questions; falls back to an in-memory graph when Neo4j is unavailable. |
| `QueryChain` | Detects multi-hop questions, generates follow-up queries (e.g. each prerequisite) and integrates the results. |
| `APIKeyManager` | Rotates across several Gemini API keys to stay within rate limits. |

Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.

### Knowledge graph

| Node | Example | Relationship | Example |
|---|---|---|---|
| `Course` | CSI106, MAD101 | `HAS_PREREQUISITE` | MAD101 → CSI106 |
| `Semester` | Semester 1, 2, … | `TAUGHT_IN` | CSI106 → Semester 1 |
| `Combo` | AI17_COM1 | `BELONGS_TO_COMBO` | AIG202c → AI17_COM1 |
| `CLO` | CSI106_CLO1 | `HAS_CLO` | CSI106 → CSI106_CLO1 |

Details: [docs/GraphRAG.md](docs/GraphRAG.md).

## Data

`Data/combined_data.json` holds the AI-major curriculum scraped from FPT University's FLM system: **45 course syllabi** with general information, learning outcomes, materials, assessments and session schedules, plus specialisation (combo) tracks. Each syllabus is split into typed documents (`general_info`, `learning_outcome_detail`, `materials`, `assessments`, `schedule`, …) before indexing.

The engine also supports a `students` section for class lists; it is intentionally left empty in this repository for privacy.

## Running

```bash
python -m venv venv && venv\Scripts\activate     # or: source venv/bin/activate
pip install -r requirements.txt

# .env — one key is enough, up to three are rotated
GEMINI_API_KEY_1=...
GEMINI_API_KEY_2=...
# optional Neo4j; otherwise an in-memory graph is used
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

python flask_app.py        # http://127.0.0.1:5000
python test_graphrag.py    # graph sanity checks
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Ask a question — body `{"message": "...", "multihop": false}` |
| `POST` | `/api/graph-query` | Hybrid vector + graph query |
| `GET` | `/api/subjects` | List indexed courses |
| `GET` | `/api/examples` | Example questions |
| `GET` | `/api/conversation` · `POST /api/conversation/clear` | Conversation history |
| `GET` | `/api/status` · `/api/graph-status` | Engine and graph health |

`/api/chat` returns the answer, the retrieved passages, multi-hop execution details and metadata (query type, courses covered, response time).

## Example questions

| Question | Route |
|---|---|
| "Bạn là ai?" | quick reply, no retrieval |
| "CSI106 là môn gì?" | single-course lookup |
| "So sánh CSI106 và MAD101" | comparative |
| "Thông tin SEG301 và các môn tiên quyết" | multi-hop: SEG301, then each prerequisite |
| "Kỳ 4 có những môn gì?" | graph traversal `Semester 4 ←TAUGHT_IN– Course` |

Simple lookups answer in a few seconds; multi-hop chains take longer because each follow-up query is retrieved and generated separately.

## Repository

```
flask_app.py            web app and REST API
advanced_rag_engine.py  router, hierarchical index, hybrid retrieval, query chain
graph_database.py       Neo4j / in-memory course graph
templates/, static/     chat UI
Data/                   curriculum data and merge scripts
json_note_extract.py    extracts per-course notes → output/
DA.ipynb                data analysis of the curriculum
```
