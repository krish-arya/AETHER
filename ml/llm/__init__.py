"""LLM financial intelligence engine (PRD section 8, Phase 3).

Turns news articles into structured, validated, source-attributed ``FinancialEvent``
records and point-in-time news features per stock.

    ingestion   news sources, de-duplication
    extraction  prompts, the analyzer, hallucination controls
    sentiment   aggregation of events into per-stock daily features
    retrieval   small local document store
    evaluation  labelled fictional cases and the metrics computed on them

See ``docs/llm-engine.md``.
"""
