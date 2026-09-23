# rss_gemma_filtering

Personalised RSS digest generator for Obsidian, powered by embedding similarity.

Every article you have ever saved in your Obsidian vault is embedded into a local
vector database. When new RSS entries arrive they are scored by cosine similarity
against that database — no LLM inference required at filtering time. The top-scoring
entries are written to a daily digest note and an interactive UMAP visualisation.

## Quick start

Start an embedding server (see [below](#embedding-server)), then:

```bash
pip install .            # installs the `rss-filter` command
rss-filter \
  --vault ~/Documents/MyVault \
  --feeds ~/Documents/MyVault/.rss-dashboard-data/data.json \
  --state-dir ~/.local/state/rss-filter
```

From a checkout, `python main.py …` is equivalent.

See [USAGE.md](USAGE.md) for the full flag reference and common workflows.

## Embedding server

Any OpenAI-compatible `/v1/embeddings` endpoint works; the recommended one is
llama.cpp's `llama-server` with EmbeddingGemma 300M (QAT, Q4_0):

```bash
llama-server -hf lmstudio-community/embeddinggemma-300m-qat-GGUF:Q4_0 \
  --host 127.0.0.1 --port 8080 \
  --embeddings --pooling mean \
  --ctx-size 2048 --batch-size 2048 --ubatch-size 2048 \
  --sleep-idle-seconds 60
```

The batch sizes must equal the context size so a whole input fits in one batch;
`--sleep-idle-seconds` unloads the model from memory between runs.

## Running on NixOS

The flake provides `packages.default` (the `rss-filter` command, tests run at build
time), a dev shell (`nix develop`: Python deps, pytest, `llama-server`) and a
home-manager module.

```nix
# flake.nix of your system configuration
inputs.rss-filter = {
  url = "github:rodgzilla/rss_gemma_filtering";
  inputs.nixpkgs.follows = "nixpkgs";
};

# in a home-manager configuration
imports = [ inputs.rss-filter.homeModules.default ];

programs.rss-filter = {
  enable = true;
  vault = "/home/me/Documents/MyVault";   # required
  feeds = "/home/me/Documents/MyVault/.rss-dashboard-data/data.json"; # required
  # stateDir defaults to "${config.xdg.stateHome}/rss-filter";
  # package defaults to this flake's packages.default.
  settings.embedding.top_k = 7;           # any config.toml key; the rest keep the packaged defaults
};
```

The module installs `rss-filter-run` (the command with `--config`, `--state-dir`,
`--vault` and `--feeds` filled in; extra arguments are passed through, e.g.
`rss-filter-run --dry-run`) and a oneshot user service without a timer: start it with
`systemctl --user start rss-filter`. The embedding server is not part of the module;
run `llama-server` as above (e.g. as a system service) at the `base_url` from the settings
(default `http://127.0.0.1:8080/v1`).

## How it works

See [EMBEDDING_PIPELINE.md](EMBEDDING_PIPELINE.md) for a detailed description of the
scoring pipeline.

## Running the tests

```bash
pytest tests/          # or: nix develop -c pytest tests/
```
