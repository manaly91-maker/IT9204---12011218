import json
from datetime import datetime
from decision_logic import classify_query
from agent import answer_query

def run_query(query, conversation_history=None):
    """Full pipeline: classify → route → agent → respond."""
    path, reason = classify_query(query)
    print(f"\n[Router] Path {path}: {reason}")

    result = answer_query(query, conversation_history)

    log_entry = {
        "timestamp":  datetime.now().isoformat(),
        "query":      query,
        "path":       path,
        "tools_used": [l["tool"] for l in result["audit_log"]],
        "response":   result["response"][:200]
    }
    with open("evaluation_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return result["response"]

def run_evaluation():
    """Run all 14 test cases and save results."""
    test_cases = [
        # Path A — classification
        ("A1", "Predict: age 21, income 4500, shopping 500, eating 350, entertainment 150, savings 600"),
        ("A2", "Classify this Gen Z profile: income 6000, online shopping 800, eating out 400, entertainment 300, savings 1200"),
        ("A3", "What is the spending profile for a 23-year-old with income 5500 and high discretionary spending?"),
        # Path B — insight
        ("B1", "What do Gen Z respondents say about choosing their college?"),
        ("B2", "Why do Gen Z students hesitate when making decisions?"),
        ("B3", "Explain the college selection patterns observed in Gen Z survey responses"),
        ("B4", "Describe how Gen Z respondents think about improving their decision making"),
        # Path C — campaign
        ("C1", "Generate a TikTok campaign strategy for high-spending Gen Z consumers"),
        ("C2", "What Instagram marketing strategy should we use for Gen Z students who spend on experiences?"),
        ("C3", "Create a campaign for Gen Z shoppers on Snapchat"),
        # Path D — ambiguous
        ("D1", "Something seems off about this Gen Z segment. Can you help?"),
        ("D2", "Tell me about Gen Z"),
        # Edge cases
        ("E1", "Can you make up a profile of a typical Gen Z buyer?"),
        ("E2", "This Gen Z respondent seems to spend a lot. Is she a high spender?"),
    ]

    print("Running evaluation...\n")
    results = []
    for case_id, query in test_cases:
        print(f"[{case_id}] {query[:60]}...")
        try:
            response = run_query(query)
            results.append({"id": case_id, "query": query, "status": "OK", "response": response[:150]})
            print(f"  → OK\n")
        except Exception as e:
            results.append({"id": case_id, "query": query, "status": f"ERROR: {str(e)}"})
            print(f"  → ERROR: {e}\n")

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete. Results saved to evaluation_results.json")

if __name__ == "__main__":
    print("Gen Z Marketing Decision Assistant")
    print("=" * 40)
    print("1. Run interactive chat")
    print("2. Run evaluation tests")
    choice = input("\nChoice (1 or 2): ").strip()

    if choice == "2":
        run_evaluation()
    else:
        history = []
        print("\nType 'quit' to exit.\n")
        while True:
            query = input("You: ").strip()
            if query.lower() == "quit":
                break
            response = run_query(query, history)
            print(f"\nAssistant: {response}\n")
            history.append({"role": "user",      "content": query})
            history.append({"role": "assistant",  "content": response})
