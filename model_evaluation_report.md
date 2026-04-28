# Model Evaluation Report: RSS Feed Filtering with LLMs

**Date:** 2026-04-27  
**Project:** `rss_gemma_filtering` — LLM-powered RSS filter integrated with Obsidian

---

## 1. Task Description

This project uses an LLM for four distinct sub-tasks, all text-only and requiring structured output:

| Sub-task | Input | Output | Est. tokens/run |
|---|---|---|---|
| **Interest profile building** | Batches of ~2,000 tokens of Obsidian notes (16K context budget) | Free-form text summary, recursively merged | ~10K input, ~2K output |
| **Pass-1 filtering** | Interest profile + up to 20 article titles | Numbered list: `N. yes / no / ?` | ~2K input, ~200 output |
| **Pass-2 filtering** | Interest profile + up to 10 titles + 300-char summaries | Numbered list: `N. yes: <reason>` | ~3K input, ~300 output |
| **Re-ranking** | Interest profile + up to 20 kept entries | Numbered list with relevance scores | ~2K input, ~200 output |

**Key requirements:**
- Reliable structured output (numbered lists, specific label formats)
- Modest context window (~4K–8K tokens per call is sufficient; 16K max for profile building)
- Text-only (no vision, audio, or multimodal inputs needed)
- Runs **daily**

**Total estimated daily token usage: ~10K input tokens, ~1K output tokens** — an extremely low-volume workload.

---

## 2. Hardware Specifications

| Component | Specification |
|---|---|
| **CPU** | Intel Core i7-7700K @ 4.20 GHz (4 cores / 8 threads) |
| **RAM** | 32 GB DDR4 |
| **GPU** | NVIDIA GeForce GTX 1070 — **8 GB GDDR5 VRAM** |
| **GPU Memory Bandwidth** | ~256 GB/s (GDDR5, slower than modern GDDR6X cards) |
| **CUDA** | 12.4 |
| **Available VRAM** | ~3.1 GB free at time of measurement (LM Studio using ~4 GB) |
| **Disk (free)** | ~50 GB |

**Notable constraints:**
- 8 GB VRAM is the hard ceiling for local inference.
- GDDR5 bandwidth limits inference speed compared to newer GPUs (RTX 3090: ~936 GB/s; GTX 1070: ~256 GB/s).
- Disk space is tight (50 GB free); large quantized models (Q4 ~8 GB per file) still fit, but leave little headroom.

---

## 3. Currently Configured Model: Gemma 4 E4B

### 3.1 Model Architecture

| Property | Value |
|---|---|
| **Model** | `google/gemma-4-E4B-it` |
| **Release** | April 2026 |
| **Architecture** | Dense transformer |
| **Effective parameters** | 4.5B (8B total including Per-Layer Embeddings) |
| **Context window** | 128K tokens |
| **Modalities** | Text, Image, Audio |
| **License** | Apache 2.0 |

Gemma 4 E4B uses **Per-Layer Embeddings (PLE)**: each decoder layer has its own embedding table, making the total weight footprint larger than the effective parameter count suggests. The full BF16 weights occupy approximately **16 GB**, well beyond this machine's VRAM.

### 3.2 VRAM Requirements by Quantization

| Quantization | Approx. VRAM | Fits GTX 1070? | Notes |
|---|---|---|---|
| **BF16 (full precision)** | ~16 GB | **No** | Exceeds VRAM by 2× |
| **Q8** | ~8 GB | **Borderline** | No headroom for KV cache; will likely OOM with current GPU usage |
| **Q4_K_M** | ~4.5–5 GB | **Yes** | ~3 GB free for KV cache; practical ceiling on this GPU |
| **Q3_K_M** | ~3.5 GB | Yes | Significant quality loss; not recommended |

LM Studio is currently consuming ~4 GB of VRAM with this model loaded, which implies it is running a **Q4 quantization**. This is consistent with the model fitting within 8 GB alongside the display server and other GPU processes (~1 GB).

### 3.3 Performance Assessment on GTX 1070

| Metric | Estimate |
|---|---|
| **Inference speed (Q4, GTX 1070)** | ~3–8 tokens/sec |
| **Time per Pass-1 batch (200 output tokens)** | ~25–65 seconds |
| **Time per full daily run** | ~3–10 minutes |
| **Quality impact of Q4** | Moderate degradation in instruction following |

The GTX 1070's GDDR5 bandwidth (~256 GB/s) is the primary bottleneck. For context, a modern RTX 3090 (936 GB/s) would produce the same Q4 model at 3–4× higher token throughput.

### 3.4 Suitability Verdict

| Criterion | Assessment |
|---|---|
| **Fits in VRAM** | Yes, at Q4 only |
| **Task capability** | Sufficient — the task is simple classification and summarization |
| **Structured output reliability** | Moderate — Q4 quantization degrades instruction-following compared to full precision |
| **Speed** | Acceptable for a daily batch job (minutes, not hours) |
| **Overkill?** | Somewhat — the multimodal capabilities (image, audio) are entirely unused |
| **Overall** | **Functional but suboptimal.** The model works, but Q4 quantization and the GTX 1070's bandwidth mean you are not getting the model's full quality. A better-matched local model or a cloud API would improve output reliability. |

---

## 4. Local Model Alternatives

All VRAM estimates assume Q4_K_M quantization. A rule of thumb: BF16 requires ~2 GB per billion parameters; Q4 requires ~0.5 GB per billion effective parameters (though PLE models like Gemma 4 E4B are heavier due to embedding tables).

### 4.1 Models That Fit Comfortably (< 6 GB Q4)

#### Gemma 3 4B (`google/gemma-3-4b-it`)
- **Params:** 4B dense | **Q4 VRAM:** ~2.5 GB | **Context:** 128K
- **Pros:** Fits easily, leaves headroom for KV cache, fast inference (~15–25 tok/s on GTX 1070)
- **Cons:** Older model, weaker instruction following than Gemma 4 E4B; lower benchmark scores
- **Verdict:** Step backward in quality from current setup; not recommended as upgrade

#### Gemma 3n E4B (`google/gemma-3n-E4B-it`)
- **Params:** 4.5B effective (like Gemma 4 E4B but older generation) | **Q4 VRAM:** ~2.5 GB | **Context:** 32K
- **Pros:** Optimized for on-device / low-resource use, fits easily
- **Cons:** 32K context window (sufficient for this task but no headroom); older generation; weaker than Gemma 4 E4B
- **Verdict:** Not an improvement over Gemma 4 E4B; skip

### 4.2 Sweet Spot (6–7.5 GB Q4)

#### Gemma 3 12B (`google/gemma-3-12b-it`) — **Recommended Local Upgrade**
- **Params:** 12B dense | **Q4 VRAM:** ~6.5–7 GB | **Context:** 128K
- **Pros:** Significantly better instruction following and structured output than any 4B model; still fits in 8 GB at Q4; 128K context
- **Cons:** Leaves ~1 GB for KV cache — tight but workable for this task's modest context sizes (~4–8K tokens/call); slower than 4B (~5–12 tok/s on GTX 1070)
- **Verdict:** **Best local option for quality.** The structured output tasks in this project (numbered label lists) benefit meaningfully from a larger model. Worth trying if pass-1/pass-2 parsing errors are observed with the current setup.

### 4.3 Benchmarked: Qwen3-30B-A3B (`qwen/qwen3-30b-a3b`)

This model was benchmarked live on 2026-04-27 against all three task types.

#### Architecture

| Property | Value |
|---|---|
| **Model** | `Qwen/Qwen3-30B-A3B` (HF) / `qwen/qwen3.6-35b-a3b` (LM Studio) |
| **Architecture** | MoE — **30.5B total / 3.3B active** parameters |
| **Layers / Experts** | 48 layers, 128 experts (8 active per token) |
| **Context window** | 32,768 tokens native (131K with YaRN RoPE scaling, non-default) |
| **License** | Apache 2.0 |

#### VRAM Analysis

| Quantization | Approx. VRAM | Fits GTX 1070? |
|---|---|---|
| **Q4_K_M** | ~16–17 GB | **No** — requires CPU offloading on this machine |

LM Studio can run this model by offloading most layers to CPU (the machine has 32 GB RAM), but this eliminates the GPU inference speed advantage. The model was confirmed running via LM Studio with CPU+GPU split offloading.

#### Thinking Mode — Critical Issue

Qwen3 defaults to **thinking mode**: chain-of-thought reasoning inside `<think>...</think>` tokens before producing the answer. Via the LM Studio OpenAI-compatible API, **thinking cannot be disabled** through either the `/no_think` prompt suffix or `extra_body={"thinking": False}`. Each call generates ~2,000–2,600 reasoning tokens before the actual answer (~27–152 tokens). This is ~98% overhead per call.

#### Benchmark Results (measured live)

| Task | Wall-clock time | Thinking tokens | Answer tokens | Total tok/s | Accuracy |
|---|---|---|---|---|---|
| Pass-1 (10 titles) | 236.6 s | 2,632 | 43 | 11.3 | **10/10 perfect** |
| Pass-2 (5 entries + summaries) | 229.0 s | 2,398 | 152 | 11.1 | **Perfect** |
| Rerank (5 entries) | 200.7 s | 2,185 | 27 | 11.0 | **Valid scores** |

**Interpretation:**
- Accuracy is excellent — Qwen3 answered all tasks correctly.
- Speed is 3–4 minutes per call, compared to ~3–30 seconds for Gemma 4 E4B Q4.
- A full daily run (490 entries, multiple passes) would take **several hours** — completely impractical.
- The ~11 tok/s total rate reflects CPU-offloaded inference; the thinking tokens alone account for ~200–230 of those seconds.

#### Potential Workaround

LM Studio may allow disabling thinking mode via a **model-level system prompt override** in the model's configuration panel (not via API parameters). If `/no_think` can be injected at the LM Studio layer before every request, effective throughput would drop to answer tokens only (~27–152 tokens/call), which would be much faster. This was not tested.

#### Verdict

| Criterion | Assessment |
|---|---|
| **Fits in VRAM** | **No** — requires CPU offload (~16–17 GB Q4) |
| **Accuracy** | Excellent — perfect on all benchmark tasks |
| **Speed (as-is)** | **Impractical** — 3–4 min/call due to thinking mode + CPU offload |
| **Speed (if thinking disabled)** | Potentially usable — needs LM Studio system prompt workaround |
| **Overall** | **Not recommended in current configuration.** Accuracy is better than Gemma 4 E4B, but the combination of no native VRAM fit and forced thinking mode makes it ~50–100× slower. Revisit if thinking can be disabled at the LM Studio config level. |

### 4.4 Models That Do Not Fit

| Model | Q4 VRAM | Reason |
|---|---|---|
| **Qwen3-30B-A3B** | ~16–17 GB | MoE total weight exceeds VRAM; requires CPU offload |
| **Gemma 3 27B** | ~15 GB | 3× over VRAM ceiling |
| **Gemma 4 26B A4B MoE** | ~14–15 GB | 26B total weights; MoE reduces compute but not memory footprint |
| **Gemma 4 31B** | ~17 GB | Far exceeds VRAM |
| **DeepSeek V4 Flash** (local) | ~50+ GB | 284B total params; cluster-scale only |
| **DeepSeek V4 Pro** (local) | ~400+ GB | 1.6T params; not consumer-feasible |

### 4.5 Local Summary Table

| Model | Q4 VRAM | Speed (est.) | Quality | Fits? | Verdict |
|---|---|---|---|---|---|
| Gemma 3 4B | ~2.5 GB | Fast | Lower | Yes | Downgrade |
| Gemma 3n E4B | ~2.5 GB | Fast | Lower | Yes | No benefit |
| **Gemma 3 12B** | ~6.5–7 GB | Moderate | **Good** | Yes (tight) | **Best local option** |
| Gemma 4 E4B | ~4.5–5 GB | Moderate | Moderate | Yes (Q4 only) | Current — functional |
| Gemma 3 27B | ~15 GB | — | Excellent | **No** | VRAM exceeded |
| Gemma 4 26B A4B MoE | ~14–15 GB | — | Excellent | **No** | VRAM exceeded |
| Qwen3-30B-A3B | ~16–17 GB | Very slow (CPU offload + thinking) | Excellent | **No** (CPU offload only) | Impractical as-is |

---

## 5. Cloud API Alternatives

### 5.1 Token Usage and Cost Model

Given the daily workload estimate (~10K input tokens, ~1K output tokens), costs are negligible for all providers. The table below uses these assumptions:

- **Input tokens/day:** 10,000
- **Output tokens/day:** 1,000
- **Days/month:** 30
- **Monthly input tokens:** 300K
- **Monthly output tokens:** 30K

### 5.2 Open/Efficient Models via OpenRouter

These models use the OpenAI-compatible API format — a **drop-in replacement** for the current LM Studio setup (same `base_url` pattern, just point to OpenRouter instead of `localhost:1234`).

| Model | Context | Input $/1M | Output $/1M | Est. monthly cost | Notes |
|---|---|---|---|---|---|
| **Gemma 3 4B** (`google/gemma-3-4b-it`) | 128K | $0.04 | $0.08 | **~$0.014** | Fast, cheap; weaker structured output |
| **Gemma 3 12B** (`google/gemma-3-12b-it`) | 128K | $0.04 | $0.13 | **~$0.016** | Best value; same quality ceiling as local 12B but at full precision |
| **Gemma 3 27B** (`google/gemma-3-27b-it`) | 128K | $0.08 | $0.16 | **~$0.029** | Excellent quality; not feasible locally |
| **Gemma 3n E4B** (`google/gemma-3n-e4b-it`) | 32K | $0.06 | $0.12 | **~$0.022** | On-device model; 32K context is sufficient but tight |

### 5.3 DeepSeek V4 (April 2026)

DeepSeek V4 was released on April 24, 2026. Both models support a 1M token context window, thinking/non-thinking modes, and the OpenAI-compatible API format.

#### DeepSeek V4 Flash (`deepseek-v4-flash`)
- **Architecture:** MoE — 284B total / **13B active** parameters
- **Context:** 1M tokens
- **Pricing:** $0.14/1M input (cache miss) · $0.0028/1M input (cache hit) · **$0.28/1M output**
- **Est. monthly cost:** ~$0.05
- **Notes:** "Flash" tier — fast, cost-effective, reasoning capabilities close to V4-Pro. Excellent for batch tasks. The 13B active params at full precision likely outperforms local Gemma 4 E4B Q4 on structured output.

#### DeepSeek V4 Pro (`deepseek-v4-pro`)
- **Architecture:** MoE — 1.6T total / **49B active** parameters
- **Context:** 1M tokens
- **Pricing (limited-time 75% off, until 2026-05-05):** $0.435/1M input · $0.87/1M output
- **Regular pricing:** $1.74/1M input · $3.48/1M output
- **Est. monthly cost (discounted):** ~$0.16 | **Regular:** ~$0.63
- **Notes:** Top-tier open-weight model; world-class reasoning. Significant overkill for RSS filtering. The 75% discount expires May 5, 2026.

### 5.4 Comparison: DeepSeek V3 0324 (via OpenRouter)

For users wanting access to DeepSeek models through OpenRouter (which aggregates multiple providers with fallback/uptime guarantees):

| Model | Context | Input $/1M | Output $/1M | Est. monthly cost |
|---|---|---|---|---|
| **DeepSeek V3 0324** (`deepseek/deepseek-chat-v3-0324`) | 163K | $0.20 | $0.77 | ~$0.08 |

Note: DeepSeek API docs state that `deepseek-chat` will be retired after July 24, 2026, routing now goes to V4-Flash.

### 5.5 Cloud API Summary Table

| Model | Provider | Est. monthly cost | Quality | Privacy | Verdict |
|---|---|---|---|---|---|
| Gemma 3 4B | OpenRouter | ~$0.014 | Moderate | Google-routed | Cheapest; modest quality |
| **Gemma 3 12B** | OpenRouter | ~$0.016 | **Good** | Google-routed | Best value cloud option |
| Gemma 3 27B | OpenRouter | ~$0.029 | Very Good | Google-routed | Good if quality matters more |
| **DeepSeek V4 Flash** | DeepSeek API | ~$0.05 | **Excellent** | Chinese company | Best quality/cost ratio |
| DeepSeek V4 Pro | DeepSeek API | ~$0.16 (disc.) / ~$0.63 | State-of-the-art | Chinese company | Overkill for this task |
| DeepSeek V3 0324 | OpenRouter | ~$0.08 | Very Good | Mixed | Good alternative route |

All costs are negligible at this usage volume. The decision is primarily about **quality** and **privacy**, not cost.

---

## 6. Privacy Considerations

| Option | Data residency | Notes |
|---|---|---|
| **Local (LM Studio)** | On-device | Full privacy; no data leaves the machine |
| **OpenRouter → Google** | US (Google Cloud) | Google's data policies apply |
| **DeepSeek API** | China | Data processed on Chinese infrastructure; relevant for sensitive personal data |
| **OpenRouter → Groq/Fireworks** | US | Third-party US providers; OpenRouter handles routing |

Given that this tool processes your personal reading interests and Obsidian notes (which constitute a detailed personal interest profile), **local inference preserves the most privacy**.

---

## 7. Recommendations

### 7.1 Stay Local (Privacy-First)

**If Gemma 4 E4B Q4 is producing reliable structured output:** Keep the current setup. It works and is fully private.

**If you observe parsing failures** (malformed numbered lists, label hallucinations): Switch to **Gemma 3 12B Q4** via LM Studio. It is a stronger instruction follower at only ~2 GB more VRAM, while still fitting within your 8 GB ceiling.

To load Gemma 3 12B in LM Studio, update `config.toml`:
```toml
[lmstudio]
model = "google/gemma-3-12b-it"  # or the LM Studio internal identifier
```

### 7.2 Move to Cloud (Quality Upgrade, Minimal Cost)

If local inference is too slow or output quality is unsatisfactory, **DeepSeek V4 Flash** is the recommended cloud option:

- Near-state-of-the-art quality (13B active parameters, full precision MoE)
- OpenAI-compatible API — **one config line change**
- ~$0.05/month at daily usage
- 1M context window (far beyond what this task requires)

Update `config.toml`:
```toml
[lmstudio]
base_url = "https://api.deepseek.com/v1"
model = "deepseek-v4-flash"
temperature = 0.1
```
Add your API key via environment variable or a secrets config.

If privacy with cloud is a concern, **Gemma 3 27B via OpenRouter** (~$0.03/month) offers excellent quality routed through US-based providers, with open weights.

### 7.3 Decision Matrix

| Priority | Recommendation |
|---|---|
| Maximum privacy | Local: Gemma 4 E4B Q4 (current) or Gemma 3 12B Q4 |
| Best local quality | **Gemma 3 12B Q4** via LM Studio |
| Best local quality (if thinking can be disabled) | Qwen3-30B-A3B via LM Studio (CPU offload — slow but accurate) |
| Best cloud quality/cost | **DeepSeek V4 Flash** via DeepSeek API |
| Best cloud, US privacy | **Gemma 3 27B** via OpenRouter |
| Absolute best quality (cost no concern) | DeepSeek V4 Pro (overkill for this task) |

---

## 8. Conclusion

Gemma 4 E4B running locally on a GTX 1070 via LM Studio is a **functional but constrained** setup for this task. The 8 GB VRAM ceiling forces Q4 quantization, which trades output quality for memory efficiency. The GTX 1070's older GDDR5 memory bus further limits inference throughput.

However, the task itself — simple structured classification with modest context requirements — does not demand a frontier model. The workload is light enough that even a 4B model performs adequately, and **the cost of cloud alternatives is essentially zero** at daily usage volumes (~$0.01–$0.05/month).

The most impactful upgrades, in order of impact:

1. **Gemma 3 12B Q4 locally** — better instruction following, same hardware, no privacy trade-off
2. **DeepSeek V4 Flash via API** — full-precision quality, negligible cost, requires trusting a Chinese API provider
3. **GPU upgrade** — an RTX 3090 (24 GB GDDR6X) would unlock Q8 precision and 3–4× faster inference for any model ≤ 24B
4. **Qwen3-30B-A3B locally (conditional)** — excellent accuracy was confirmed in benchmarks, but the mandatory thinking mode (CPU offload + 2,000–2,600 thinking tokens/call) makes it ~50–100× slower than Gemma 4 E4B. Only viable if thinking can be disabled via LM Studio's model configuration.

---

*Report generated by OpenCode on 2026-04-27.*
