# UC3 — automotive ADAS agentic RAG pipeline

A working implementation of the architecture described in Section 2 of the
paper (knowledge construction → prompt templates → ReAct agentic RAG),
instantiated for the automotive driver-behaviour / pedestrian-crossing ADAS
use case (Figure 5 / UC3).

## What's real vs. placeholder

This is a genuine, tested pipeline — the ReAct loop, tool routing, TF-IDF
retrieval, and safeguard checks all run and are covered by
`tests/test_agent_end_to_end.py`. What's **not** included, for two reasons,
is the actual regulatory text:

1. **Copyright/scope** — the AI Act and GDPR are long documents; this repo
   ships short paraphrased placeholder summaries in `data/*/SAMPLE_*.txt`
   purely so the pipeline has something to index and you can see it work.
2. **Network** — this environment can't reach eur-lex.europa.eu, so the real
   documents need to be fetched from your own machine.

**Before using this for anything real**, replace the `SAMPLE_*.txt` files
with the actual consolidated texts. `knowledge_construction/sources_manifest.csv`
lists the official EUR-Lex URLs for each domain folder. Any plain-text
export works, as long as article/annex headers stay on their own line in
one of these forms:

```
Article 10
Annex IV
Regulation (EU) 2019/2144, Article 4
```

The chunker (`knowledge_construction/build_knowledge_base.py`) splits on
those headers automatically.

## Install

```bash
pip install -r requirements.txt
```

## Run the tests (no API key needed)

```bash
python -m pytest tests/ -v
```

This proves the retrieval + ReAct loop machinery works using a scripted
`MockLLM`, independent of any actual language model.

## Run UC3 for real

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-...
python examples/uc3_automotive_adas.py
```

or with OpenAI:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python examples/uc3_automotive_adas.py
```

Add `--mock` to run the same example against the scripted mock instead
(useful for CI, or to see the expected shape of a run without spending API
credits).

The script prints the full Thought/Action/Observation/Final Answer trace and
saves it to `examples/uc3_trace.json` — that JSON is what you'd feed into a
figure-rendering step to regenerate a Figure-5-style image from a live run
instead of a hand-authored one.

## Extending to other domains

Add a new folder under `data/`, list it in
`knowledge_construction/sources_manifest.csv`, and reference its folder name
as a `domain` value in agent tool calls — `agent/tools.py` and
`agent/react_agent.py` are already domain-agnostic.

## Swapping in a real embedding-based / LightRAG backend

`DomainIndex` in `knowledge_construction/build_knowledge_base.py` currently
uses TF-IDF (zero extra infra, good enough to prove correctness). If you
want to match the paper's LightRAG-based retrieval exactly, swap the
`TfidfVectorizer` + cosine-similarity search in `DomainIndex` for a LightRAG
index — `SearchTool` in `agent/tools.py` only depends on `DomainIndex.search()`
returning a list of `{"article_ref", "text", "score"}` dicts, so the rest of
the pipeline doesn't need to change.
