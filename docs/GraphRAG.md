# Hybrid GraphRAG

FPTU RAG migrated gradually from vector-only retrieval to **hybrid GraphRAG**:

- **Vector search** (FAISS) for semantic similarity
- **Graph traversal** (Neo4j, or an in-memory graph) for relationships between courses
- A **hybrid pipeline** that merges both before answer generation

## Pipeline

```
User query
    ↓
Query processing ── entity extraction (course codes, semesters, combos) + intent analysis
    ↓                         ↓
Vector search            Graph traversal
(FAISS embeddings)       (Neo4j / in-memory entities)
    ↓                         ↓
Result integration ── merge, score boosting, de-duplication
    ↓
Answer generation ── hybrid context, relationship-aware response
```

## Setup

```bash
pip install neo4j flask sentence-transformers
```

`.env`:

```env
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
GEMINI_API_KEY_3=your_key_3

# optional — the in-memory graph is used when Neo4j is unavailable
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

Run Neo4j locally with Docker if you want a real graph database:

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

## Usage

```python
from advanced_rag_engine import AdvancedRAGEngine

rag_engine = AdvancedRAGEngine(api_keys, enable_graph=True)
rag_engine.initialize("Data/combined_data.json")

# vector-only
print(rag_engine.query("CSI106 là môn gì?")["answer"])

# hybrid vector + graph
result = rag_engine.hybrid_graph_query("CSI106 và các môn liên quan")
print(result["answer"])
print(result["metadata"]["vector_results_count"], result["metadata"]["graph_results_count"])
print(result["metadata"]["entities_extracted"])
```

REST: `POST /api/graph-query` with `{"message": "..."}`, and `GET /api/graph-status`.

## Graph schema

| Node | Description | Properties |
|---|---|---|
| `Course` | A course (CSI106, MAD101, …) | name, credits, semester, description |
| `Semester` | A term in the programme | number, year |
| `Combo` | Specialisation track | name, short_name, track_type |
| `CLO` | Course learning outcome | clo_id, details, course_code |

| Relationship | Meaning | Example |
|---|---|---|
| `HAS_PREREQUISITE` | prerequisite course | MAD101 -[HAS_PREREQUISITE]-> CSI106 |
| `TAUGHT_IN` | taught in a semester | CSI106 -[TAUGHT_IN]-> Semester_1 |
| `BELONGS_TO_COMBO` | part of a track | AIG202c -[BELONGS_TO_COMBO]-> AI17_COM1 |
| `HAS_CLO` | has a learning outcome | CSI106 -[HAS_CLO]-> CSI106_CLO1 |

## What the graph adds

| Question | Vector only | With the graph |
|---|---|---|
| "Learning path from CSI106 to AIG202c" | isolated facts about each course | the prerequisite chain CSI106 → MAD101 → CSD203 → AIG202c, explained |
| "Courses related to CSI106" | semantically similar text | dependent courses, same-semester courses and same-track courses |
| "Which courses are in semester 4?" | text search | a complete list via `Semester_4 ←TAUGHT_IN– Course` |

## Graceful degradation

1. Neo4j available → full GraphRAG
2. Neo4j unavailable → in-memory graph entities
3. Graph extraction fails → vector-only
4. Graph query error → vector search fallback

## Example trace

```
========== HYBRID GRAPHRAG QUERY ==========
USER QUERY: 'CSI106 và các môn liên quan'
STEP 1: Vector search for semantic similarity...
STEP 2: Entity extraction for graph traversal... ['CSI106']
STEP 3: Graph traversal found 3 relationship results
STEP 4: Integrated results: 5 vector + 3 graph = 7 unique
STEP 5: Generating enhanced answer...
  - Processing time: 2.34s
========== HYBRID GRAPHRAG COMPLETED ==========
```

## Roadmap

- [x] Hybrid vector + graph architecture
- [x] Entity extraction and relationship discovery
- [ ] Full Neo4j-backed deployment
- [ ] Graph algorithms (PageRank, community detection)
- [ ] Natural language → Cypher
- [ ] Incremental graph updates

## Troubleshooting

- *"Neo4j driver is not installed"* → `pip install neo4j`, or keep running in vector-only mode.
- *GraphRAG is slower than vector-only* — entity extraction, traversal and integration add work. Cache entity extraction, limit traversal depth, or process asynchronously.
