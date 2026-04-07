import sys

from process_rps_comments import canonicalize_comment

TEST_CASES = [
    ("Rock", "rock"),
    (" rock ", "rock"),
    ("STONE", "rock"),
    ("paper", "paper"),
    ("Paper 📄", "paper"),
    ("papet", "paper"),
    ("papper", "paper"),
    ("scissor", "scissors"),
    ("scissors", "scissors"),
    ("sissors", "scissors"),
    ("scissot", "scissors"),
    ("rock 🪨", "rock"),
    ("🪨 rock", "rock"),
    ("maruthi paper", "paper"),
    ("phone", "noise"),
    ("noise", "noise"),
    ("rock paper", "ambiguous"),
    ("paper scissors", "ambiguous"),
]

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    passed = 0

    print("Simulated Comment Normalization Test Results")
    print("-" * 70)

    for raw, expected in TEST_CASES:
        result = canonicalize_comment(raw)
        predicted = result["canonical_label"]
        ok = predicted == expected
        if ok:
            passed += 1

        print(f"RAW: {raw!r}")
        print(f"  advanced_normalized: {result['advanced']!r}")
        print(f"  predicted: {predicted}")
        print(f"  expected : {expected}")
        print(f"  method   : {result['method']}")
        print(f"  status   : {'PASS' if ok else 'FAIL'}")
        print()

    print(f"Passed {passed}/{len(TEST_CASES)} tests")

if __name__ == "__main__":
    main()
