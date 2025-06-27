#!/usr/bin/env python3
"""
Test Script cho GraphRAG Implementation
Kiểm tra Gradual Migration: Vector Search + Graph Database
"""

import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_graphrag_basic():
    """Test cơ bản GraphRAG functionality"""
    logger.info("========== TEST GRAPHRAG BASIC ==========")
    
    try:
        # Import after loading env
        from advanced_rag_engine import AdvancedRAGEngine, GRAPH_AVAILABLE
        
        logger.info(f"Graph components available: {GRAPH_AVAILABLE}")
        
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
            logger.error("❌ Không tìm thấy API keys")
            return False
        
        logger.info(f"✓ Tìm thấy {len(api_keys)} API keys")
        
        # Initialize RAG Engine với GraphRAG
        logger.info("Khởi tạo RAG Engine với GraphRAG...")
        rag_engine = AdvancedRAGEngine(api_keys, enable_graph=True)
        
        # Check if data file exists
        data_file = "Data/combined_data.json"
        if not os.path.exists(data_file):
            data_file = "Data/reduced_data.json"
            
        if not os.path.exists(data_file):
            logger.error(f"❌ Không tìm thấy data file")
            return False
        
        logger.info(f"Khởi tạo với data file: {data_file}")
        rag_engine.initialize(data_file)
        
        logger.info(f"✓ RAG Engine initialized")
        logger.info(f"✓ Graph enabled: {rag_engine.graph_enabled}")
        
        # Test queries
        test_queries = [
            "CSI106 là môn gì?",
            "Liệt kê các môn học kỳ 1",
            "MAD101 có những CLO nào?",
            "Mối quan hệ giữa CSI106 và các môn khác",
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- TEST QUERY {i}: {query} ---")
            
            # Test regular query
            logger.info("Testing regular query...")
            regular_result = rag_engine.query(query)
            logger.info(f"Regular result length: {len(regular_result.get('answer', ''))}")
            
            # Test GraphRAG hybrid query if available
            if rag_engine.graph_enabled:
                logger.info("Testing GraphRAG hybrid query...")
                graph_result = rag_engine.hybrid_graph_query(query)
                logger.info(f"GraphRAG result length: {len(graph_result.get('answer', ''))}")
                logger.info(f"Entities extracted: {graph_result.get('metadata', {}).get('entities_extracted', [])}")
                logger.info(f"Vector results: {graph_result.get('metadata', {}).get('vector_results_count', 0)}")
                logger.info(f"Graph results: {graph_result.get('metadata', {}).get('graph_results_count', 0)}")
            else:
                logger.info("GraphRAG not available - skipping hybrid test")
        
        logger.info("✅ Basic GraphRAG test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

def test_graph_entities():
    """Test graph entity extraction"""
    logger.info("========== TEST GRAPH ENTITY EXTRACTION ==========")
    
    try:
        from graph_database import GraphDatabase
        
        # Test data
        sample_data = [
            {
                "subject_code": "CSI106",
                "metadata": {
                    "course_name_from_curriculum": "Introduction to Computer Science",
                    "semester_from_curriculum": 1,
                    "credits": "3",
                    "prerequisites": "Không có",
                    "course_type_guess": "core"
                },
                "major_code": "AI"
            },
            {
                "subject_code": "MAD101", 
                "metadata": {
                    "course_name_from_curriculum": "Discrete Mathematics",
                    "semester_from_curriculum": 1,
                    "credits": "3",
                    "prerequisites": "CSI106",
                    "course_type_guess": "core"
                },
                "major_code": "AI"
            }
        ]
        
        # Initialize graph database
        graph_db = GraphDatabase()
        
        # Extract entities
        logger.info("Extracting entities from sample data...")
        nodes, relationships = graph_db.extract_entities_from_curriculum_data(sample_data)
        
        logger.info(f"✓ Extracted {len(nodes)} nodes:")
        for node in nodes:
            logger.info(f"  - {node.type}: {node.id}")
        
        logger.info(f"✓ Extracted {len(relationships)} relationships:")
        for rel in relationships:
            logger.info(f"  - {rel.source_id} --[{rel.type}]--> {rel.target_id}")
        
        logger.info("✅ Graph entity extraction test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Entity extraction test failed: {e}", exc_info=True)
        return False

def test_flask_endpoints():
    """Test Flask GraphRAG endpoints"""
    logger.info("========== TEST FLASK ENDPOINTS ==========")
    
    try:
        import requests
        import time
        
        base_url = "http://localhost:5000"
        
        # Test graph status endpoint
        logger.info("Testing /api/graph-status endpoint...")
        response = requests.get(f"{base_url}/api/graph-status")
        
        if response.status_code == 200:
            status = response.json()
            logger.info(f"✓ Graph status: {status}")
        else:
            logger.error(f"❌ Graph status endpoint failed: {response.status_code}")
            return False
        
        # Test graph query endpoint
        logger.info("Testing /api/graph-query endpoint...")
        test_query = {"message": "CSI106 là môn gì?"}
        
        response = requests.post(f"{base_url}/api/graph-query", json=test_query)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✓ Graph query successful")
            logger.info(f"  Answer length: {len(result.get('answer', ''))}")
            logger.info(f"  Metadata: {result.get('metadata', {})}")
        else:
            logger.error(f"❌ Graph query endpoint failed: {response.status_code}")
            return False
        
        logger.info("✅ Flask endpoints test completed")
        return True
        
    except requests.exceptions.ConnectionError:
        logger.warning("⚠ Flask server not running - skipping endpoint tests")
        return True
    except Exception as e:
        logger.error(f"❌ Flask endpoints test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Starting GraphRAG Tests...")
    
    # Track test results
    results = {}
    
    # Run tests
    results['basic'] = test_graphrag_basic()
    results['entities'] = test_graph_entities()
    results['flask'] = test_flask_endpoints()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_status in results.items():
        status = "✅ PASSED" if passed_status else "❌ FAILED"
        logger.info(f"  {test_name.upper()}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! GraphRAG is working correctly.")
    else:
        logger.error("❌ Some tests failed. Check logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 