"""
pheno_agent — An agentic system for celiac disease diagnosis.

Inspired by the DeepRare multi-agent architecture, this system uses
specialised agents (Data Gatherer, Signal Extractor, Critic, Adjudicator)
orchestrated by a central host to diagnose celiac disease from EHR notes
and TTG-IgA lab values.  All LLM inference runs locally via Ollama.
"""
