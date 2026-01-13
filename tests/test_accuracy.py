import sys
import os

# Add the parent directory to path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import get_query_engine

# --- THE GOLDEN SET ---
# Facts found in Nvidia 2025 10-K (Page 34)
TEST_CASES = [
    {
        "question": "What was the revenue for Fiscal Year 2025?",
        "expected_keywords": ["130.497", "130.5", "billion"]
    },
    {
        "question": "What was the income tax expense for fiscal years 2025 and 2024?",
        "expected_keywords": ["11.1","4.1","billion"]
    },
    {
        "question": "What were the depreciation expenses for fiscal years 2025, 2024, and 2023?",
        "expected_keywords": ["1.3", "894", "844", "billion", "million"] 
    }
]

def run_tests():
    print("⏳ Loading Engine for Testing...")
    engine = get_query_engine()
    
    print(f"\n🚀 Running {len(TEST_CASES)} Verification Tests...\n")
    
    passed = 0
    for i, test in enumerate(TEST_CASES):
        q = test["question"]
        expected = test["expected_keywords"]
        
        print(f"Test #{i+1}: {q}")
        response = str(engine.query(q)).lower() # Convert to lowercase for easier matching
        
        # Check if ALL keywords are in the answer
        missing = [word for word in expected if word.lower() not in response]
        
        if not missing:
            print("✅ PASS")
            passed += 1
        else:
            print(f"❌ FAIL")
            print(f"   Expected to find: {expected}")
            print(f"   Actual Answer: {response[:100]}...") # Show first 100 chars
        print("-" * 30)

    print(f"\n🏁 Result: {passed}/{len(TEST_CASES)} Tests Passed.")

if __name__ == "__main__":
    run_tests()