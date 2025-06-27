# GraphRAG Implementation for FPTU RAG

## Tổng quan

Đây là implementation **Gradual Migration** từ Vector-only RAG sang **Hybrid GraphRAG** theo kiến trúc trong tài liệu nghiên cứu. Hệ thống kết hợp:

- ✅ **Vector Database** (FAISS) cho semantic search
- ✅ **Graph Database** (Neo4j) cho relationship traversal
- ✅ **Hybrid Query Pipeline** kết hợp cả hai

## Kiến trúc Hybrid GraphRAG

```
User Query
    ↓
┌─────────────────────────────────────┐
│        Query Processing             │
│  - Entity Extraction               │
│  - Intent Analysis                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────┐  ┌─────────────────┐
│  Vector Search  │  │ Graph Traversal │
│  (Semantic)     │  │ (Relationships) │
│                 │  │                 │
│  FAISS Index    │  │  Neo4j/Memory   │
│  Embeddings     │  │  Entities       │
└─────────────────┘  └─────────────────┘
    ↓                       ↓
┌─────────────────────────────────────┐
│      Result Integration             │
│  - Vector + Graph Results          │
│  - Score Boosting                  │
│  - Deduplication                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Enhanced Answer Generation      │
│  - Hybrid Context                  │
│  - Relationship-aware Responses    │
└─────────────────────────────────────┘
```

## Cài đặt và Setup

### 1. Dependencies

```bash
pip install neo4j flask sentence-transformers
```

### 2. Environment Variables

Tạo file `.env`:

```env
# API Keys
GEMINI_API_KEY_1=your_key_1
GEMINI_API_KEY_2=your_key_2
GEMINI_API_KEY_3=your_key_3

# Neo4j (optional - fallback to in-memory if not available)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Neo4j Setup (Optional)

Nếu muốn sử dụng Neo4j thực sự:

```bash
# Download và start Neo4j
# Hoặc sử dụng Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

## Sử dụng

### 1. Khởi tạo với GraphRAG

```python
from advanced_rag_engine import AdvancedRAGEngine

# Enable GraphRAG mode
rag_engine = AdvancedRAGEngine(api_keys, enable_graph=True)
rag_engine.initialize("Data/combined_data.json")
```

### 2. Vector-only Query (cũ)

```python
result = rag_engine.query("CSI106 là môn gì?")
print(result['answer'])
```

### 3. Hybrid GraphRAG Query (mới)

```python
result = rag_engine.hybrid_graph_query("CSI106 là môn gì?")
print(result['answer'])
print(f"Vector results: {result['metadata']['vector_results_count']}")
print(f"Graph results: {result['metadata']['graph_results_count']}")
print(f"Entities: {result['metadata']['entities_extracted']}")
```

### 4. Flask API Endpoints

#### Regular Chat

```bash
POST /api/chat
{
    "message": "CSI106 là môn gì?",
    "multihop": false
}
```

#### GraphRAG Query

```bash
POST /api/graph-query
{
    "message": "CSI106 và các môn liên quan"
}
```

#### Graph Status

```bash
GET /api/graph-status
```

## Knowledge Graph Schema

### Entities (Nodes)

| Type       | Description                   | Properties                           |
| ---------- | ----------------------------- | ------------------------------------ |
| `Course`   | Môn học (CSI106, MAD101, ...) | name, credits, semester, description |
| `Semester` | Kỳ học (1, 2, 3, ...)         | number, year                         |
| `Combo`    | Chuyên ngành hẹp              | name, short_name, track_type         |
| `CLO`      | Course Learning Outcomes      | clo_id, details, course_code         |

### Relationships (Edges)

| Type               | Description         | Example                                 |
| ------------------ | ------------------- | --------------------------------------- |
| `HAS_PREREQUISITE` | Môn tiên quyết      | MAD101 -[HAS_PREREQUISITE]-> CSI106     |
| `TAUGHT_IN`        | Học trong kỳ        | CSI106 -[TAUGHT_IN]-> Semester_1        |
| `BELONGS_TO_COMBO` | Thuộc combo         | AIG202c -[BELONGS_TO_COMBO]-> AI17_COM1 |
| `HAS_CLO`          | Có learning outcome | CSI106 -[HAS_CLO]-> CSI106_CLO1         |

## Test và Debugging

### 1. Chạy Test Script

```bash
python test_graphrag.py
```

### 2. Check Graph Status

```python
# In Python
if rag_engine.graph_enabled:
    print("GraphRAG enabled")
    if hasattr(rag_engine, 'graph_entities'):
        entities = rag_engine.graph_entities
        print(f"Nodes: {len(entities['nodes'])}")
        print(f"Relationships: {len(entities['relationships'])}")
```

### 3. Flask Development Mode

```bash
python flask_app.py
# Truy cập http://localhost:5000/api/graph-status
```

## Lợi ích của GraphRAG

### 1. Multi-hop Reasoning

**Trước (Vector-only):**

```
Query: "Lộ trình học từ CSI106 đến AIG202c"
→ Chỉ tìm được thông tin riêng lẻ về từng môn
```

**Sau (GraphRAG):**

```
Query: "Lộ trình học từ CSI106 đến AIG202c"
→ Tìm được path: CSI106 → MAD101 → CSD203 → AIG202c
→ Giải thích prerequisite chain và dependencies
```

### 2. Relationship Discovery

**Trước:**

```
Query: "Các môn liên quan đến CSI106"
→ Chỉ tìm theo semantic similarity
```

**Sau:**

```
Query: "Các môn liên quan đến CSI106"
→ Tìm được:
  - Môn phụ thuộc vào CSI106 (through HAS_PREREQUISITE)
  - Môn cùng kỳ (through TAUGHT_IN)
  - Môn cùng combo (through BELONGS_TO_COMBO)
```

### 3. Contextual Understanding

**Trước:**

```
Query: "Kỳ 4 có những môn gì?"
→ Vector search through text content
```

**Sau:**

```
Query: "Kỳ 4 có những môn gì?"
→ Graph traversal: Semester_4 -[TAUGHT_IN]- Course nodes
→ Structured, complete list với relationships
```

## Fallback Mechanism

System được thiết kế với **graceful degradation**:

1. **Neo4j available** → Full GraphRAG mode
2. **Neo4j unavailable** → In-memory graph entities
3. **Graph extraction fails** → Vector-only fallback
4. **Graph query errors** → Vector search backup

## Monitoring và Logs

Logs chi tiết cho debugging:

```
========== HYBRID GRAPHRAG QUERY ==========
USER QUERY: 'CSI106 và các môn liên quan'
STEP 1: Vector Search for semantic similarity...
STEP 2: Entity extraction for graph traversal...
Extracted entities from query: ['CSI106']
STEP 3: Graph traversal...
Graph traversal với 1 entities...
Graph traversal found 3 relationship results
STEP 4: Integrating vector and graph results...
Integrated results: 5 vector + 3 graph = 7 unique
STEP 5: Generating enhanced answer...
HYBRID GraphRAG SUMMARY:
  - Vector results: 5
  - Graph results: 3
  - Integrated results: 7
  - Entities extracted: ['CSI106']
  - Processing time: 2.34s
========== HYBRID GRAPHRAG COMPLETED ==========
```

## Roadmap

- [x] **Phase 1**: Basic hybrid architecture (Vector + Graph)
- [x] **Phase 2**: Entity extraction và relationship discovery
- [ ] **Phase 3**: Neo4j integration với actual database
- [ ] **Phase 4**: Advanced graph algorithms (PageRank, community detection)
- [ ] **Phase 5**: Natural language to Cypher query translation
- [ ] **Phase 6**: Dynamic graph updates và real-time learning

## Troubleshooting

### Graph không enable

```python
# Check log messages:
# "Neo4j driver chưa được cài đặt - pip install neo4j"
# "Neo4j không khả dụng - fallback to vector-only mode"

# Solution:
pip install neo4j
# Hoặc chạy trong vector-only mode bình thường
```

### Performance issues

```python
# GraphRAG có thể chậm hơn vector-only do:
# 1. Entity extraction step
# 2. Graph traversal computation
# 3. Result integration

# Optimization:
# - Cache entity extraction results
# - Limit graph traversal depth
# - Use async processing
```

## Kết luận

GraphRAG implementation này cung cấp foundation vững chắc cho việc nâng cấp từ vector-only sang relationship-aware RAG system. Với gradual migration approach, bạn có thể:

1. **Giữ nguyên functionality hiện tại** (vector search)
2. **Thêm graph capabilities** khi cần thiết
3. **Fallback gracefully** khi graph unavailable
4. **Scale up** đến full Neo4j khi ready

Đây là bước đầu tiên hướng tới intelligent search engine có khả năng multi-hop reasoning và contextual understanding như mô tả trong tài liệu nghiên cứu.
