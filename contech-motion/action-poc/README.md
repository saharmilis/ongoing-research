# Fixed-label video action classification — POC

Classify the three original 15-second recordings using a small multimodal model and a finite label set. The first test asks one question about each full clip; it does not yet generate chapters or process a full day.

**Current execution status:** input preparation and local tests passed. **No model classifications have been produced in this environment.** The initial d3 model-weight download was blocked by the network proxy; the subsequent retry was not executed because automatic approval review hit a usage limit. The latest observed allowlist still omitted the download CDN `us.aws.cdn.hf.co`. There is no configured GPU here. Prepared result JSONs explicitly contain `label: null` and `prepared_only_no_model_inference`; they are not predictions.

The model-loading and forward-pass adapters follow the pinned source implementations, but are **not yet validated end to end with their weights**. Run the commands below on your machine and return the result JSONs. A reported inference exception is a failed test, never an `other` prediction.

## Start with d3-nano

[d3-nano](https://huggingface.co/vllm-sr/d3-nano) is the direct match for this experiment: its native `system_one` API accepts video, choice criteria, and returns a score for each option without generating text.

[LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) is a generative video-language model. Here it is adapted to a choice task: disable thinking, assign one-token letters to the options, and score the next-token logits in one forward pass. No prose or chain of thought is generated. This is an experimental adapter, not a separately decision-trained LFM checkpoint.

| Backend | Pinned model revision | Required Transformers |
| --- | --- | --- |
| d3 | `8198c9f763b046605b2f4827b4999971fa2c92c4` | `5.17.0` |
| lfm | `71d8e745d97c224984e0f20412f532a5ce9dfd3e` | `5.21.0` |

Use **separate virtual environments** because the model cards specify different Transformers versions. d3 uses Apache-2.0; LFM weights use the LFM Open License v1.0 linked from its model card. No model weights or third-party model implementation files are committed here.

## GPU run: basic four-label test

From the repository root, create the d3 environment. Python 3.12 is the local preparation/test environment.

```bash
cd contech-motion/action-poc
python -m venv .venv-d3
source .venv-d3/bin/activate

# Install PyTorch + torchvision appropriate for your CUDA/driver first,
# using https://pytorch.org/get-started/locally/.
# Then install the remaining dependencies:
pip install -r requirements-d3.txt

python classify_actions.py \
  --model d3 \
  --device auto \
  --output results/d3-basic.json
```

`auto` uses CUDA when available, otherwise CPU. Both adapters use BF16 weights; CUDA execution requires BF16 support. A 16GB BF16-capable GPU is a conservative starting point for these small tests, not a measured minimum. CPU execution is available with `--device cpu`, but has not been benchmarked. Plan enough disk for several GB of weights per model. The model is loaded once and reused across the three clips.

For CPU-specific PyTorch installation before requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-d3.txt
```

Then test LFM in its own environment:

```bash
deactivate
python -m venv .venv-lfm
source .venv-lfm/bin/activate

# Install the appropriate PyTorch/torchvision build first as above.
pip install -r requirements-lfm.txt

python classify_actions.py \
  --model lfm \
  --device auto \
  --output results/lfm-basic.json
```

Weights and trusted model code are fetched from the pinned revisions. The script explicitly uses `trust_remote_code=True`, required by the publisher implementations. Model download access must include the actual CDN redirects as well as `huggingface.co`. For a local debug traceback, add `--debug`; avoid publishing download URLs or credentials from debug logs.

After both complete:

```bash
python summarize_results.py results/d3-basic.json results/lfm-basic.json
```

Send the **two JSON files** back for review. They include scores for every label, timing, model revision, sampling provenance, and native response details. The summary is just for quick comparison.

## No-model input check

This mode requires only the existing motion pipeline dependencies, not Torch or Transformers:

```bash
pip install -r ../requirements.txt
python classify_actions.py --model d3 --prepare-only \
  --output results/my-prepared-inputs.json
```

The checked-in [d3 preparation report](results/prepared-d3.json) and [LFM preparation report](results/prepared-lfm.json) cover all three actual recordings. Each contains 30 chronological frames from 0 to 14.96 seconds, at prepared dimensions 512 × 384, with source SHA-256 values. There are no fabricated labels.

## Labels and definitions

Default: `--label-set basic`.

| ID | Viewer label | Evidence required |
| --- | --- | --- |
| `standing_still` | Standing Still | Vehicle is stationary and the attachment performs no meaningful operation; ignore minor vibration and foliage |
| `driving` | Driving | Vehicle translation supported by changing viewpoint/parallax; moving the bucket alone is insufficient |
| `digging` | Digging | Visible excavation, scraping, or scooping; attachment movement alone is insufficient |
| `other` | Other | Unclear, unsupported, or mixed activity without a dominant class |

The optional `--label-set expanded` adds:

| ID | Viewer label | Evidence required |
| --- | --- | --- |
| `loading_or_dumping` | Loading / Dumping | Visible material transfer or release into a truck, container, or pile |
| `tool_repositioning` | Tool Repositioning | Boom/arm/bucket movement without clear digging or material transfer |

Start with the four-label test, then repeat with the expanded set if `Other` is common or digging is confused with repositioning. Keep the taxonomy fixed across test runs and record which set was used. A human label such as “digging episode” can cover substeps where the visible action is only repositioning; resolve that labeling definition before deciding a model needs more parameters.

Additional categories such as grading or compacting should come from observed baseline footage and equipment types. Do not introduce untestable labels solely because they are plausible construction actions.

## Inputs, evidence, and fairness

Defaults are `standing-1.mp4`, `standing-2.mp4`, and `digging-1.mp4` from the parent folder. Override with `--input-dir` and `--videos`.

```bash
python classify_actions.py --model d3 --input-dir /path/to/clips \
  --videos a.mp4 b.mp4 c.mp4 --output results/custom.json
```

- Native temporal video input, not a single still or a contact sheet.
- Approximately 2 Hz, capped at 32 evenly distributed frames. For the 15s sources: 30 frames from start to end.
- d3 uses its native video decoding and published preprocessing. LFM uses the same sampled source indices, RGB frames resized to at most 512 pixels on the long side, and explicit source timestamps. Each model still applies its own native normalization/tokenization.
- Do not feed 1–24h recordings directly to this POC: a fixed 32-frame cap would miss most events. Use short windows for later chapter analysis.
- Filenames, human expected labels, and motion scores are excluded from the question text. Filepaths are passed only to the video decoder/native media API.
- The three clip names are not model predictions. The expected human baselines are two standing clips and one digging clip, but those expectations never enter the prompt.
- To inspect option-order sensitivity, repeat a completed test with `--reverse-options` and a different output path. LFM can be sensitive to letter/order effects; d3 has its own publisher-defined option handling.

## Output and uncertainty

Each completed clip has `label`, `option_scores`, `top_score`, `top_two_margin`, `review_suggested`, `inference_seconds`, input metadata, and `raw_model_response`.

These option distributions are **not calibrated real-world accuracy probabilities**. d3's decision readout and LFM's softmax over allowed letter logits are different scoring mechanisms. Compare labels and observed errors first, not confidence magnitudes across models.

The LFM adapter also reports `allowed_letter_probability_mass`: if extremely small, the model preferred some output format outside the forced choices, so a confident conditional label is less persuasive. Check that diagnostic rather than assuming constrained decoding guarantees a valid interpretation.

A margin below 0.15 is an **unvalidated review flag only**; the script never changes a predicted class to `Other` based on that flag. No abstention thresholds are selected from these three clips. A tie remains reviewable. Errors, unsupported inputs, and unfinished calls have explicit statuses and null labels.

Results are saved after each completed video. If a later call fails, earlier predictions remain intact. The CLI exits nonzero on model setup/inference failure. The source video cannot be overwritten by the output JSON.

Timing includes preprocessing and model inference for each clip; CUDA is synchronized before and after. Model-load time is reported separately. There are no latency claims yet. Finite labels reduce output generation, but the visual encoder and model prefill still perform substantial computation.

## Evaluation and iteration

1. Run d3 with the basic labels on the three baseline clips.
2. Run LFM with the same basic labels and compare exact labels, distributions, ambiguity, and latency.
3. For ambiguous results, inspect sampled temporal evidence, label definitions, and optionally the expanded taxonomy.
4. Add driving, empty attachment movement, loading/dumping, obstructed/poor-light footage, and unfamiliar activities. Include genuine `Other` examples.
5. Evaluate on separate clips/cameras before selecting an abstention policy or a deployment model.
6. Only then decide whether failures need better sampling, domain examples, changed definitions, fine-tuning, or a larger model.

Three clips are an initial sanity check, not a reliable estimate of accuracy or evidence that Driving/Other works.

For later chapters: classify short overlapping windows, preserve timestamps and option scores, then merge stable labels with transition persistence. Avoid one VLM call per second. The existing motion graph can schedule extra analysis around transitions, but it must not override these initial blind tests. `Other` review should sample diverse clips; frequent unknowns can reflect blur or ambiguity as well as a missing action class.

## Tests and current verification boundary

```bash
pip install pytest
python -m pytest -q test_classify_actions.py
```

Local tests validate the three real input videos, chronological frame counts/timestamps, finite-label score handling, no expected-label leakage in the state, and prepare-only CLI behavior. These do **not** prove either model's inference adapter works with downloaded weights. That remains the next concrete check on the GPU machine.

## Primary documentation read

- [d3-nano model card and native choice/video examples](https://huggingface.co/vllm-sr/d3-nano).
- [d3 pinned runtime](https://huggingface.co/vllm-sr/d3-nano/blob/8198c9f763b046605b2f4827b4999971fa2c92c4/d3_runtime.py): decision format, video sampling, CPU fallback, and fixed BF16 loading.
- [LFM3-VL-2B model card](https://huggingface.co/LiquidAI/LFM3-VL-2B).
- [LFM pinned chat template](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/71d8e745d97c224984e0f20412f532a5ce9dfd3e/chat_template.jinja): `enable_thinking=False`.
- [LFM pinned forward implementation](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/71d8e745d97c224984e0f20412f532a5ce9dfd3e/modeling_lfm3_vl.py): `logits_to_keep` support used for next-letter scoring.
