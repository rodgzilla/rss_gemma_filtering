# rss_gemma_filtering

Personalised RSS digest generator for Obsidian, powered by embedding similarity.

Every article you have ever saved in your Obsidian vault is embedded into a local
vector database. When new RSS entries arrive they are scored by cosine similarity
against that database — no LLM inference required at filtering time. The top-scoring
entries are written to a daily digest note and an interactive UMAP visualisation.

## Quick start

```bash
pip install -r requirements.txt
python main.py \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/subscriptions.opml
```

See [USAGE.md](USAGE.md) for the full flag reference and common workflows.

## How it works

See [EMBEDDING_PIPELINE.md](EMBEDDING_PIPELINE.md) for a detailed description of the
scoring pipeline.

## Running the tests

```bash
pytest tests/
```
