# FPTU Syllabus RAG System

## Overview

FPTU Syllabus RAG is a powerful Retrieval Augmented Generation system that enables students and faculty to query detailed information about FPT University syllabi using natural language. The system combines advanced vector search technology with contextual query understanding to provide accurate, relevant responses drawn directly from official course syllabi.

## Features

- **Natural Language Queries**: Ask questions about courses in everyday language
- **Contextual Understanding**: System recognizes entities, relationships, and attributes in your queries
- **Automatic Course Detection**: Identifies which course you're asking about, even without explicit mention
- **Multi-Course Support**: Search across all courses or filter to specific subjects
- **Semantic Search**: Find information based on meaning, not just keywords
- **Source References**: View the exact syllabus sections used to generate answers
- **Vietnamese Language Support**: Fully supports queries and responses in Vietnamese

## System Architecture

The system implements a modern RAG (Retrieval Augmented Generation) architecture:

1. **Data Crawling & Processing**: Extracts syllabus data from FPT's FLM system
2. **Text Chunking**: Divides syllabi into semantic chunks with rich metadata
3. **Vector Embedding**: Transforms text chunks into mathematical vectors using multilingual transformers
4. **Indexing**: Creates efficient FAISS indexes for fast similarity searching
5. **Contextual Query Processing**: Interprets natural language queries and identifies entities
6. **Retrieval**: Finds the most relevant syllabus chunks using semantic search
7. **Generation**: Uses Google's Gemini models to generate human-friendly answers from retrieved contexts

## Getting Started

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended for faster processing)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/FPTU_RAG.git
   cd FPTU_RAG
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   python -m playwright install
   ```

3. Set up environment variables by creating a .env file:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

### Data Preparation Pipeline

1. **Crawl syllabus data**:

   ```bash
   python Syllabus_crawler/syllabus_crawler.py
   ```

2. **Process and chunk the data**:

   ```bash
   python syllabus_chunker.py
   ```

3. **Generate embeddings**:

   ```bash
   python generate_embeddings.py
   ```

4. **Create FAISS index**:
   ```bash
   python create_faiss_index.py
   ```

### Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Then open your browser at http://localhost:8501 to access the application.

## Usage Examples

Here are some example queries you can try:

- "DPL302m là môn học gì?"
- "Môn Deep Learning có bao nhiêu tín chỉ?"
- "Chuẩn đầu ra của môn DPL là gì?"
- "Session 45 học gì?"
- "Có bao nhiêu môn toán?"
- "Đưa tất cả link Coursera môn DPL302m"
- "Môn DPL302m có bao nhiêu bài kiểm tra, cách tính điểm thế nào?"

## Project Structure

```
FPTU_RAG/
├── Syllabus_crawler/            # Crawler component
│   ├── syllabus_crawler.py      # Main crawler script
│   └── FPT FLM Syllabus Crawler.md  # Documentation
├── Chunk/                       # Chunked data storage
│   └── enhanced_chunks.json     # Processed chunks with metadata
├── Embedded/                    # Embedding storage
│   └── all_embeddings.json      # Vector embeddings for chunks
├── Faiss/                       # FAISS index storage
│   ├── all_syllabi_faiss.index  # Main FAISS index
│   └── index_info.json          # Index metadata
├── Entity/                      # Entity data storage
│   └── entity_data.json         # Extracted entity information
├── syllabus_chunker.py          # Chunking processor
├── generate_embeddings.py       # Embedding generator
├── create_faiss_index.py        # FAISS index creation
├── contextual_query_processor.py  # Query processing system
├── app.py                       # Streamlit application
└── README.md                    # Project documentation
```

## Technologies Used

- **Playwright**: For web crawling and data extraction
- **Sentence Transformers**: For generating text embeddings
- **FAISS**: For efficient similarity search
- **Google Gemini AI**: For answer generation
- **Streamlit**: For the web interface
- **Python**: Core programming language

## Limitations and Notes

- The system only provides information that exists in the crawled syllabi
- Query understanding works best with Vietnamese questions about syllabus content
- Processing speed depends on your hardware, especially for embedding generation and FAISS search

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FPT University for the syllabus data structure
- Sentence Transformers team for the embedding models
- Google for Gemini API access
