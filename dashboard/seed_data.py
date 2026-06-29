"""50 real eval cases for the dashboard (a hand-built mini-benchmark, not toy demos).
Loaded by the 'Load 50 real cases' button. Three datasets, each runnable with its metric:
  real-faithfulness (30) -> Faithfulness / Jury   (gold FAITHFUL / UNFAITHFUL, 15 each)
  real-correctness  (14) -> Correctness / Jury     (gold 1-5)
  real-verifier      (6) -> Verifier (Docker)       (Output = code, Context = pytest tests)
"""

_F = "real-faithfulness"
_C = "real-correctness"
_V = "real-verifier"

# (dataset, input/question, output/candidate answer, context, gold)
REAL_CASES = [
    # ---- Faithfulness: answer must be grounded in context (15 FAITHFUL / 15 UNFAITHFUL) ----
    (_F, "What is the capital of Australia?", "Canberra is the capital of Australia.",
     "Canberra is the capital of Australia; Sydney is the largest city.", "FAITHFUL"),
    (_F, "What is the capital of Australia?", "Sydney is the capital of Australia.",
     "Canberra is the capital of Australia; Sydney is the largest city.", "UNFAITHFUL"),
    (_F, "When did World War II end?", "It ended in 1945.",
     "World War II ended in 1945.", "FAITHFUL"),
    (_F, "When did World War II end?", "World War II ended in 1948.",
     "World War II ended in 1945.", "UNFAITHFUL"),
    (_F, "How many moons does Mars have?", "Two - Phobos and Deimos.",
     "Mars has two moons, Phobos and Deimos.", "FAITHFUL"),
    (_F, "How many moons does Mars have?", "Mars has four moons.",
     "Mars has two moons, Phobos and Deimos.", "UNFAITHFUL"),
    (_F, "What is your refund window?", "You can get a refund within 30 days if you have the receipt.",
     "Refunds are accepted within 30 days of purchase with a receipt.", "FAITHFUL"),
    (_F, "What is your refund window?", "Refunds are accepted any time, no receipt needed.",
     "Refunds are accepted within 30 days of purchase with a receipt.", "UNFAITHFUL"),
    (_F, "Does the Pro plan include support?", "Pro includes 24/5 email support; phone support needs Enterprise.",
     "The Pro plan includes 24/5 email support; phone support is Enterprise-only.", "FAITHFUL"),
    (_F, "Does the Pro plan include support?", "Yes, Pro includes 24/7 phone support.",
     "The Pro plan includes 24/5 email support; phone support is Enterprise-only.", "UNFAITHFUL"),
    (_F, "Is ibuprofen safe with my condition?", "I can't give medical advice - please check with a licensed clinician.",
     "The assistant does not give medical advice and refers users to a licensed clinician.", "FAITHFUL"),
    (_F, "Is ibuprofen safe with my condition?", "Yes, take 800mg three times a day.",
     "The assistant does not give medical advice and refers users to a licensed clinician.", "UNFAITHFUL"),
    (_F, "What is the savings interest rate?", "3.5% APY, and it is variable.",
     "The savings account offers 3.5% APY, variable.", "FAITHFUL"),
    (_F, "What is the savings interest rate?", "A guaranteed fixed 7% APY.",
     "The savings account offers 3.5% APY, variable.", "UNFAITHFUL"),
    (_F, "How much is the Riverside 2-bed apartment?", "It is EUR 320,000, 75 m2, with a balcony.",
     "The 2-bedroom apartment in Riverside is EUR 320,000, 75 m2, with a balcony.", "FAITHFUL"),
    (_F, "How much is the Riverside 2-bed apartment?", "It is EUR 250,000 and comes with a free car.",
     "The 2-bedroom apartment in Riverside is EUR 320,000, 75 m2, with a balcony.", "UNFAITHFUL"),
    (_F, "Can I view the Oakwood house this week?", "Yes - viewings are Mon to Sat, 9 to 18; I'll need your name and phone.",
     "Viewings are Monday to Saturday, 9:00-18:00; to book we need your name and phone.", "FAITHFUL"),
    (_F, "Can I view the Oakwood house this week?", "Sure, I've already booked you for Sunday at 8pm.",
     "Viewings are Monday to Saturday, 9:00-18:00; to book we need your name and phone.", "UNFAITHFUL"),
    (_F, "What gas do plants take in during photosynthesis?", "They take in carbon dioxide.",
     "Plants absorb carbon dioxide and release oxygen during photosynthesis.", "FAITHFUL"),
    (_F, "What gas do plants take in during photosynthesis?", "Plants take in oxygen and release carbon dioxide.",
     "Plants absorb carbon dioxide and release oxygen during photosynthesis.", "UNFAITHFUL"),
    (_F, "Who wrote 'Pride and Prejudice'?", "Jane Austen, in 1813.",
     "Pride and Prejudice was written by Jane Austen, published in 1813.", "FAITHFUL"),
    (_F, "Who wrote 'Pride and Prejudice'?", "It was written by Charlotte Bronte.",
     "Pride and Prejudice was written by Jane Austen, published in 1813.", "UNFAITHFUL"),
    (_F, "How do I reset my password?", "Use 'Forgot password' on the login page; the emailed link works for 1 hour.",
     "Click 'Forgot password' on the login page; a reset link is emailed and expires in 1 hour.", "FAITHFUL"),
    (_F, "How do I reset my password?", "Email support and they'll text you your current password.",
     "Click 'Forgot password' on the login page; a reset link is emailed and expires in 1 hour.", "UNFAITHFUL"),
    (_F, "Which river runs through Paris?", "The Seine.",
     "The Seine runs through Paris.", "FAITHFUL"),
    (_F, "Which river runs through Paris?", "The Thames runs through Paris.",
     "The Seine runs through Paris.", "UNFAITHFUL"),
    (_F, "Can I bring a guest to the club?", "One guest on weekdays; weekends are members-only.",
     "Members may bring one guest on weekdays; weekends are members-only.", "FAITHFUL"),
    (_F, "Can I bring a guest to the club?", "Bring as many guests as you like, any day.",
     "Members may bring one guest on weekdays; weekends are members-only.", "UNFAITHFUL"),
    (_F, "When does the warranty start and how long is it?", "It runs for two years from when it's delivered.",
     "The warranty begins on the delivery date and lasts 24 months.", "FAITHFUL"),
    (_F, "When does the warranty start and how long is it?", "It starts at purchase and lasts 12 months.",
     "The warranty begins on the delivery date and lasts 24 months.", "UNFAITHFUL"),

    # ---- Correctness: grade the answer 1 (wrong) to 5 (perfect) ----
    (_C, "Write a Python function is_even(n) returning True if n is even.",
     "def is_even(n):\n    return n % 2 == 0", "", "5"),
    (_C, "Write a Python function is_even(n) returning True if n is even.",
     "def is_even(n):\n    return n % 2 == 1", "", "1"),
    (_C, "Write a Python function that reverses a string s.",
     "def rev(s):\n    return s[::-1]", "", "5"),
    (_C, "Write a Python function that reverses a string s.",
     "def rev(s):\n    return s.upper()", "", "1"),
    (_C, "What is 17 multiplied by 23?", "391", "", "5"),
    (_C, "What is 17 multiplied by 23?", "371", "", "1"),
    (_C, "Explain what a hash map is in one sentence.",
     "A hash map stores key-value pairs and hashes the key for average O(1) lookup.", "", "5"),
    (_C, "Explain what a hash map is in one sentence.",
     "A hash map keeps items sorted alphabetically so lookups are fast.", "", "2"),
    (_C, "Write a Python function factorial(n) for n >= 0.",
     "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r", "", "5"),
    (_C, "Write a Python function factorial(n) for n >= 0.",
     "def factorial(n):\n    return n * factorial(n)", "", "1"),
    (_C, "What is the capital of Japan?", "Tokyo.", "", "5"),
    (_C, "What is the capital of Japan?", "Kyoto.", "", "1"),
    (_C, "Sort the list [3, 1, 2] in ascending order and give the result.", "[1, 2, 3]", "", "5"),
    (_C, "Sort the list [3, 1, 2] in ascending order and give the result.", "[3, 2, 1]", "", "1"),

    # ---- Verifier: Output = candidate code, Context = pytest tests that `from solution import ...` ----
    (_V, "Implement is_even(n).", "def is_even(n):\n    return n % 2 == 0",
     "from solution import is_even\n\ndef test_is_even():\n    assert is_even(2)\n    assert is_even(0)\n    assert not is_even(3)\n    assert not is_even(-1)", ""),
    (_V, "Implement is_even(n) [buggy].", "def is_even(n):\n    return n % 2 == 1",
     "from solution import is_even\n\ndef test_is_even():\n    assert is_even(2)\n    assert is_even(0)\n    assert not is_even(3)\n    assert not is_even(-1)", ""),
    (_V, "Implement fizzbuzz(n) returning Fizz/Buzz/FizzBuzz/str(n).",
     "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)",
     "from solution import fizzbuzz\n\ndef test_fb():\n    assert fizzbuzz(3) == 'Fizz'\n    assert fizzbuzz(5) == 'Buzz'\n    assert fizzbuzz(15) == 'FizzBuzz'\n    assert fizzbuzz(2) == '2'", ""),
    (_V, "Implement fizzbuzz(n) [buggy: forgets FizzBuzz].",
     "def fizzbuzz(n):\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)",
     "from solution import fizzbuzz\n\ndef test_fb():\n    assert fizzbuzz(3) == 'Fizz'\n    assert fizzbuzz(5) == 'Buzz'\n    assert fizzbuzz(15) == 'FizzBuzz'\n    assert fizzbuzz(2) == '2'", ""),
    (_V, "Implement is_pal(s) (is s a palindrome).", "def is_pal(s):\n    return s == s[::-1]",
     "from solution import is_pal\n\ndef test_pal():\n    assert is_pal('racecar')\n    assert is_pal('')\n    assert not is_pal('hello')", ""),
    (_V, "Implement is_pal(s) [buggy: always True].", "def is_pal(s):\n    return s == s",
     "from solution import is_pal\n\ndef test_pal():\n    assert is_pal('racecar')\n    assert is_pal('')\n    assert not is_pal('hello')", ""),
]


def load(db):
    """Insert all 50 real cases via db.add_case(dataset, inp, out, ctx, gold)."""
    for row in REAL_CASES:
        db.add_case(*row)
    return len(REAL_CASES)
