#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_rag_engine import AdvancedRAGEngine

def test_student_final():
    """Final test for student queries with high boost factors"""
    
    url = "http://localhost:5000/api/chat"
    
    test_cases = [
        {
            "query": "danh sách sinh viên AI",
            "expected": "student_overview should be top result"
        },
        {
            "query": "sinh viên ngành AI", 
            "expected": "student data should appear in results"
        },
        {
            "query": "thông tin sinh viên DE170004",
            "expected": "specific student detail"
        }
    ]
    
    print("=== FINAL STUDENT TEST WITH HIGH BOOST FACTORS ===")
    
    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        expected = test["expected"]
        
        print(f"\n{i}. Testing: '{query}'")
        print(f"Expected: {expected}")
        print("-" * 50)
        
        payload = {
            "message": query,
            "multihop": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                
                print(f"✅ SUCCESS ({elapsed:.1f}s)")
                print(f"Answer length: {len(answer)} chars")
                
                # Check for student-related content
                student_indicators = [
                    'DE170004', 'DE170061', 'Hoàng Trung Quân', 'Trần Ngọc Thiện',
                    'danh sách sinh viên', 'sinh viên ngành AI', '15 sinh viên'
                ]
                
                found_indicators = [ind for ind in student_indicators if ind in answer]
                
                if found_indicators:
                    print(f"🎯 STUDENT DATA FOUND: {found_indicators}")
                    print(f"Answer preview: {answer[:200]}...")
                else:
                    print(f"❌ NO STUDENT DATA: {answer[:200]}...")
                    
            else:
                print(f"❌ ERROR {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"⏱️ TIMEOUT after 30s")
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")

def test_coursera_fix():
    """Test fix cho missing DWP301c trong Coursera queries"""
    
    print("=" * 80)
    print("TEST COURSERA FIX - DWP301c MISSING BUG")
    print("=" * 80)
    
    # Initialize engine
    engine = AdvancedRAGEngine("dummy-key")  # Sử dụng dummy key cho test
    engine.initialize("Data/combined_data.json")
    
    # Test query
    query = "Môn coursera kì 5"
    print(f"\nTesting query: '{query}'")
    print("-" * 50)
    
    try:
        # Get search intent and config để debug
        intent = engine.query_router.analyze_query(query)
        print(f"Query intent: {intent.query_type}")
        print(f"Target subjects: {intent.target_subjects}")
        
        # Get search configuration
        config = engine._get_search_config(intent, query.lower())
        print(f"Search config: {config}")
        print(f"Coursera boost enabled: {config.get('coursera_boost', False)}")
        
        # Test semantic search directly để debug
        print(f"\nDirect semantic search test:")
        semantic_results = engine._semantic_search(query, config)
        print(f"Semantic search returned: {len(semantic_results)} results")
        
        # Check for DWP301c in semantic results
        for i, res in enumerate(semantic_results[:20], 1):
            subject_code = res.get('subject_code', 'N/A')
            score = res.get('score', 0)
            course_type = res.get('metadata', {}).get('course_type_guess', 'N/A')
            if subject_code == 'DWP301c' or course_type.startswith('coursera'):
                print(f"  {i}. {subject_code} (score: {score:.2f}, type: {course_type})")
        
        # Perform search strategy to get raw results
        raw_results = engine._search_strategy(query, intent)
        print(f"\nRaw search results: {len(raw_results)}")
        
        # Check for DWP301c in raw results
        dwp_found = False
        coursera_courses = []
        
        print(f"\nALL RESULTS (not just top 10):")
        for i, res in enumerate(raw_results, 1):
            subject_code = res.get('subject_code', 'N/A')
            score = res.get('score', 0)
            course_type = res.get('metadata', {}).get('course_type_guess', 'N/A')
            print(f"{i}. {subject_code} (score: {score:.2f}, type: {course_type})")
            
            if subject_code == 'DWP301c':
                dwp_found = True
                print(f"   ✅ DWP301c FOUND at position {i}!")
                
            if course_type.startswith('coursera'):
                coursera_courses.append(subject_code)
        
        print(f"\nCoursera courses found: {coursera_courses}")
        print(f"DWP301c found: {'✅ YES' if dwp_found else '❌ NO'}")
        
        # Also test specific semester filtering
        semester_5_courses = []
        for res in raw_results:
            metadata = res.get('metadata', {})
            if metadata.get('semester_from_curriculum') == 5:
                semester_5_courses.append(res.get('subject_code', 'N/A'))
        
        print(f"Semester 5 courses found: {semester_5_courses}")
        
        # FINAL TEST: Search specifically for DWP301c or Web Development
        print(f"\n=== FINAL DEBUG: Search for 'DWP301c Web Development Python' ===")
        dwp_specific_results = engine._semantic_search("DWP301c Web Development Python", config)
        print(f"DWP-specific search returned: {len(dwp_specific_results)} results")
        
        for i, res in enumerate(dwp_specific_results[:10], 1):
            subject_code = res.get('subject_code', 'N/A')
            score = res.get('score', 0)
            course_type = res.get('metadata', {}).get('course_type_guess', 'N/A')
            if subject_code == 'DWP301c':
                print(f"  ✅ {i}. DWP301c FOUND! (score: {score:.2f}, type: {course_type})")
            else:
                print(f"  {i}. {subject_code} (score: {score:.2f}, type: {course_type})")
        
        # Check if DWP301c exists in processed data
        print(f"\n=== CHECK: DWP301c in processed data ===")
        dwp_items = [item for item in engine.data if item.get('subject_code') == 'DWP301c']
        print(f"DWP301c items in processed data: {len(dwp_items)}")
        if dwp_items:
            for item in dwp_items:
                print(f"  - Type: {item.get('type')}")
                print(f"  - Course type: {item.get('metadata', {}).get('course_type_guess', 'N/A')}")
                print(f"  - Semester: {item.get('metadata', {}).get('semester_from_curriculum', 'N/A')}")
                print(f"  - Content preview: {item.get('content', '')[:100]}...")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_student_final()
    test_coursera_fix() 