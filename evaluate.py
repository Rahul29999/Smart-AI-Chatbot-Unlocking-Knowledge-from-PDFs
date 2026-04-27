
from chatbot import extract_pdf_text, preprocess_text, build_index, search_query

test_cases = [
    {"query": "What is this document about?",         "expected_source": "pdf"},
    {"query": "Explain quantum computing in detail",   "expected_source": "gpt"},
    {"query": "What are the main topics covered?",    "expected_source": "pdf"},
    {"query": "Tell me about artificial intelligence", "expected_source": "gpt"},
]

def run_evaluation(pdf_path: str) -> float:
    text = extract_pdf_text(pdf_path)
    sentences = preprocess_text(text)
    index, _ = build_index(sentences)
    correct = 0
    print(f"\n{'Query':<45} {'Expected':<10} {'Got':<10} {'Result'}")
    print("-" * 80)
    for case in test_cases:
        _, source = search_query(case["query"], sentences, index)
        ok = source == case["expected_source"]
        if ok:
            correct += 1
        print(f"{case['query']:<45} {case['expected_source']:<10} {source:<10} {'✓ PASS' if ok else '✗ FAIL'}")
    accuracy = correct / len(test_cases) * 100
    print(f"\nRouting Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    return accuracy
