# Part 2: Resume ingestion + RAG: what we built and why

(Rewrite in your own words. Questions to answer:)
- Why redact PII before calling Gemini, and where do the real contact details come back? (Export, Part 7)
- Why chunk by bullet instead of fixed-size text windows?
- Why does each chunk's embedding text include its project/company name?
- Why local fastembed instead of an API for embeddings? (privacy, no rate limits, no PyTorch)
- Why passage_embed vs query_embed for BGE models?
- Why is cosine similarity the same as a dot product here? (vectors are normalized)
- Why a numpy store instead of Chroma/FAISS for ~50 vectors? When WOULD you switch?
- Why store embedding_id with the index? (vectors from different models are not comparable)
- Why validate the session id with a strict regex? (path traversal)
- Why verbatim bullets in the schema? (Part 6 verifier compares tailored text against them)
