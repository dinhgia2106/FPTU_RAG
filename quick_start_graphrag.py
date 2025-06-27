#!/usr/bin/env python3
"""
Quick Start Script cho GraphRAG Demo
Chạy nhanh để test hybrid functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def demo_graphrag():
    """Demo GraphRAG capabilities"""
    
    print("🚀 FPTU GraphRAG Demo - Gradual Migration")
    print("=" * 50)
    
    try:
        # Import components
        from advanced_rag_engine import AdvancedRAGEngine, GRAPH_AVAILABLE
        
        print(f"📊 Graph components available: {'✅' if GRAPH_AVAILABLE else '❌'}")
        
        # Get API keys
        api_keys = []
        for i in range(1, 4):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                api_keys.append(key)
        
        if not api_keys:
            original_key = os.getenv("GEMINI_API_KEY")
            if original_key:
                api_keys.append(original_key)
        
        if not api_keys:
            print("❌ Không tìm thấy API keys trong .env file")
            print("   Cần thiết lập GEMINI_API_KEY hoặc GEMINI_API_KEY_1")
            return False
        
        print(f"🔑 API Keys: {len(api_keys)} keys loaded")
        
        # Initialize RAG Engine
        print("\n🔄 Khởi tạo GraphRAG Engine...")
        rag_engine = AdvancedRAGEngine(api_keys, enable_graph=True)
        
        # Find data file
        data_files = ["Data/combined_data.json", "Data/reduced_data.json"]
        data_file = None
        
        for file in data_files:
            if os.path.exists(file):
                data_file = file
                break
        
        if not data_file:
            print("❌ Không tìm thấy data file")
            print("   Cần có Data/combined_data.json hoặc Data/reduced_data.json")
            return False
        
        print(f"📁 Data file: {data_file}")
        
        # Initialize with data
        print("⚙️ Đang khởi tạo với dữ liệu...")
        rag_engine.initialize(data_file)
        
        print(f"✅ Engine initialized")
        print(f"📈 Graph enabled: {'✅' if rag_engine.graph_enabled else '❌'}")
        
        if hasattr(rag_engine, 'graph_entities'):
            entities = rag_engine.graph_entities
            print(f"🧮 Graph entities extracted:")
            print(f"   - Nodes: {len(entities.get('nodes', []))}")
            print(f"   - Relationships: {len(entities.get('relationships', []))}")
        
        # Demo queries
        print("\n" + "=" * 50)
        print("🎯 DEMO QUERIES")
        print("=" * 50)
        
        demo_queries = [
            {
                "question": "CSI106 là môn gì?",
                "description": "Basic subject inquiry"
            },
            {
                "question": "CSI106 và các môn liên quan",
                "description": "Relationship discovery"
            },
            {
                "question": "Liệt kê môn học kỳ 1",
                "description": "Semester-based query"
            }
        ]
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n--- DEMO {i}: {demo['description']} ---")
            print(f"Query: {demo['question']}")
            
            # Test vector-only
            print("\n🔍 Vector-only RAG:")
            try:
                vector_result = rag_engine.query(demo['question'])
                vector_answer = vector_result.get('answer', '')
                print(f"Length: {len(vector_answer)} chars")
                print(f"Results: {len(vector_result.get('search_results', []))}")
                
                # Show first 150 chars of answer
                preview = vector_answer[:150] + "..." if len(vector_answer) > 150 else vector_answer
                print(f"Preview: {preview}")
                
            except Exception as e:
                print(f"❌ Vector query failed: {e}")
            
            # Test GraphRAG if available
            if rag_engine.graph_enabled:
                print("\n🕸️ Hybrid GraphRAG:")
                try:
                    graph_result = rag_engine.hybrid_graph_query(demo['question'])
                    graph_answer = graph_result.get('answer', '')
                    metadata = graph_result.get('metadata', {})
                    
                    print(f"Length: {len(graph_answer)} chars")
                    print(f"Vector results: {metadata.get('vector_results_count', 0)}")
                    print(f"Graph results: {metadata.get('graph_results_count', 0)}")
                    print(f"Entities found: {metadata.get('entities_extracted', [])}")
                    
                    # Show first 150 chars of answer
                    preview = graph_answer[:150] + "..." if len(graph_answer) > 150 else graph_answer
                    print(f"Preview: {preview}")
                    
                    # Compare with vector-only
                    if len(graph_answer) > len(vector_answer):
                        print("✨ GraphRAG answer is more comprehensive!")
                    
                except Exception as e:
                    print(f"❌ GraphRAG query failed: {e}")
            else:
                print("\n🔄 GraphRAG not available - using vector fallback")
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("=" * 50)
        
        print("\n📚 Next steps:")
        print("1. Test Flask app: python flask_app.py")
        print("2. Try API endpoints: /api/graph-query")
        print("3. Check documentation: GraphRAG_README.md")
        print("4. Install Neo4j for full graph functionality")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main function"""
    success = demo_graphrag()
    
    if not success:
        print("\n💡 Troubleshooting:")
        print("- Check .env file has API keys")
        print("- Ensure Data/combined_data.json exists")
        print("- Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n🚀 GraphRAG is ready to use!")

if __name__ == "__main__":
    main() 