"""RAGAS smoke - detect a hallucination with the Faithfulness metric, judged by a
LOCAL LM Studio model, zero cost.

Faithfulness = is the answer grounded in the retrieved context (vs made up)? It is
THE core RAG-quality check: a faithful answer scores ~1, a hallucinated one ~0.

Gotcha (real packaging bug): ragas 0.4.3 hard-imports
`langchain_community.chat_models.vertexai.ChatVertexAI`, a path removed from newer
langchain-community. We never use Vertex, so we register a tiny stub module before
importing ragas so the import resolves. (Alternative: pin compatible versions.)

Run: cd ~/projects/my-unsloth-finetune && source env-wsl.sh \
       && evals/ragas-demo/.venv/bin/python evals/ragas-demo/ragas_demo.py
"""

import asyncio
import os
import sys
import types

# --- stub the removed langchain path so ragas 0.4.3 imports (we never use Vertex) ---
_vx = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:  # noqa: E701 - stub only; ragas only imports the symbol
    pass
_vx.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vx

from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import SingleTurnSample  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import Faithfulness  # noqa: E402

JUDGE = os.getenv("RAGAS_JUDGE", "phi-4-14b")
llm = LangchainLLMWrapper(ChatOpenAI(
    model=JUDGE,
    base_url=os.environ["LM_STUDIO_API_BASE"],
    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
    temperature=0,
))
metric = Faithfulness(llm=llm)
print(f"judge: {JUDGE} via {os.environ['LM_STUDIO_API_BASE']}\n")

CONTEXT = ["The Eiffel Tower is located in Paris, France, and was completed in 1889."]
QUESTION = "Where is the Eiffel Tower and when was it completed?"
SAMPLES = [
    ("faithful", "The Eiffel Tower is in Paris and was completed in 1889."),
    ("hallucinated", "The Eiffel Tower is in Berlin and was completed in 1950."),
]

print("=== RAGAS Faithfulness (grounded in context => ~1, made up => ~0) ===")
scores = {}
for label, answer in SAMPLES:
    s = SingleTurnSample(user_input=QUESTION, response=answer, retrieved_contexts=CONTEXT)
    score = asyncio.run(metric.single_turn_ascore(s))
    scores[label] = score
    print(f"  {label:13s} faithfulness={score:.2f}  ('{answer[:50]}...')")

print(f"\nDISCRIMINATION: faithful({scores['faithful']:.2f}) "
      f"{'>' if scores['faithful'] > scores['hallucinated'] else '<='} "
      f"hallucinated({scores['hallucinated']:.2f}) -> "
      f"{'PASS' if scores['faithful'] > scores['hallucinated'] else 'FAIL'}")
