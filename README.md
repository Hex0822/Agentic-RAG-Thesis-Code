# Process Pipeline

This folder contains the main processing pipeline for the project. The workflow is
structured as clear, extendable steps:

1. **claim_extraction**: split an input claim into atomic sub-claims.
2. **keyword_generation**: generate search keywords from each sub-claim (GPT-5.2).
3. **keyword_search**: search each keyword with Tavily and keep top articles.
4. **text_processing**: coarse clean -> coref -> sentence split -> rule filter.
5. **relevance_ranking**: rank candidate sentences for each sub-claim with MiniLM cross-encoder.
6. **evidence_packaging**: keep per-sub-claim top-k ranked sentences and attach local context
   (previous/current/next sentence) for downstream LLM calls.
7. **retrieval_quality**: use GPT-5.2 as an agentic retrieval controller to judge evidence,
   output provisional conclusion, and choose next action (`FINALIZE_SUBCLAIM`,
   `REFINE_KEYWORDS`, or `STOP_SUBCLAIM`); if refine, run another focused retrieval round.

## Structure

- `claim_extraction.py`: step 1 implementation.
- `keyword_generation.py`: step 2 implementation.
- `search.py`: step 3 implementation (Tavily search + article normalization).
- `text_processing.py`: step 4 implementation (cleanup + coref + sentence filtering).
- `relevance_ranking.py`: step 5 implementation (MiniLM relevance ranking).
- `evidence_packaging.py`: step 6 implementation (top-k evidence payload with context).
- `retrieval_quality.py`: step 7 implementation (agentic conclusion + next-action decision + keyword refinement plan).
- `pipeline.py`: LangGraph pipeline wiring the steps together.
- `config.py`: shared configuration and model setup.

## Quick start

Run the demo pipeline:

```bash
python process/run.py
```
