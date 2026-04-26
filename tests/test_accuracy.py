# import sys
# import os

# # Add the parent directory to path so we can import 'src'
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.rag_engine import get_query_engine

# # --- THE GOLDEN SET ---
# # Facts found in Nvidia 2025 10-K (Page 34)
# TEST_CASES = [
#     {
#         "question": "What was the revenue for Fiscal Year 2025?",
#         "expected_keywords": "130.5 billion"
#     }
# ]

# def run_tests():
#     print("⏳ Loading Engine for Testing...")
#     engine = get_query_engine()
    
#     print(f"\n🚀 Running {len(TEST_CASES)} Verification Tests...\n")
    
#     passed = 0
#     for test in enumerate(TEST_CASES):
#         response = str(engine.query(test["question"])).lower() # Convert to lowercase for easier matching

#         print("Q:",test["question"])
#         print("A:", response)
#         print("Expected:",  test["expected"])
#         print("---")

# if __name__ == "__main__":
#     run_tests()



import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import get_query_engine
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- TEST CASES ---
TEST_CASES = [
    {
        "question": "What was the revenue for Fiscal Year 2025?",
        "ground_truth": "130.5 billion"
    },
    {
        "question": "What were the depreciation expenses in 2025?",
        "ground_truth": "around 1.3 billion"
    }
]


def evaluate_faithfulness(llm, context, answer):
    prompt = f"""
    You are a strict evaluator.
    Given the CONTEXT and ANSWER, determine if the answer is fully supported by the context.

    CONTEXT:
    {context}

    ANSWER:
    {answer}

    Respond with ONLY one word: YES or NO
    """
    return str(llm.complete(prompt)).strip().upper()

def evaluate_correctness(llm, question, answer, ground_truth):
    prompt = f"""
You are an evaluator.
Compare the model answer with the ground truth.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
MODEL ANSWER: {answer}

Is the answer correct? Respond with ONLY YES or NO.
"""
    return str(llm.complete(prompt)).strip().upper()



def run_tests():
    engine = get_query_engine()

    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY for evaluation")
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.0
    )

    print(f"Running {len(TEST_CASES)} LLM-Based Evaluation Tests...")

    passed_correct = 0
    passed_faithful = 0

    for i, test in enumerate(TEST_CASES):
        q = test["question"]
        gt = test["ground_truth"]

        response = engine.query(q)
        answer = str(response)
        # Collect retrieved context
        context = "".join([node.node.text for node in response.source_nodes])

        correctness = evaluate_correctness(llm, q, answer, gt)
        faithfulness = evaluate_faithfulness(llm, context, answer)

        print(f"Test #{i+1}")
        print("Q:", q)
        print("Answer:", answer[:200])
        print("Correctness:", correctness)
        print("Faithfulness:", faithfulness)
        print("-" * 40)

        if correctness == "YES":
            passed_correct += 1
        if faithfulness == "YES":
            passed_faithful += 1

    print(f"FINAL RESULTS:")
    print(f"Correct Answers: {passed_correct}/{len(TEST_CASES)}")
    print(f"Faithful Answers (No Hallucination): {passed_faithful}/{len(TEST_CASES)}")

if __name__ == "__main__":
    run_tests()

