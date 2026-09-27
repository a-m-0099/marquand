# Local Jev design

Date: 2026-09-26

## Intent

Run a Jev-equivalent ("System One" typed-decision model, TypeSafe AI) entirely on this laptop,
with feature parity to the hosted API and comparable accuracy and latency. Used for: text
classification/routing, agent decisions (browser steps, shell-command safety), decisions over
screenshots/images, realtime loops (NPCs in a Godot game), and ad-hoc realtime
questions.

Constraints: HTTP API drop-in plus a CLI, on-demand server, existing LM Studio models first and our own
trained model where those fall short, context window on the high end without overworking the laptop.
Single user, localhost only, one model loaded per server process.

## Hardware

Ryzen 7 7840HS (16 threads), 14 GB RAM, RX 7700S 8 GB (Navi33, gfx1102, Vulkan/RADV),
Radeon 780M iGPU. LM Studio models in `~/.lmstudio/models`.

## Contract (parity with api.typesafe.ai, from the official SDK's OpenAPI models)

- `POST /v1/systemone` body `{state, model, questions}`.
  - `state`: string | object | array.
  - `questions`: map name → one of
    - `noul`: `instructions?`, `criteria?: {true?, false?}` → `{type, noul}`
    - `choice`: `instructions?`, `criteria: {option: description|null}` (1–255) →
      `{type, choice, probabilities, confidence}`
    - `score`: `instructions?`, `criteria: [level, …]` (2–10, ordered, index = score) →
      `{type, score, legend, probabilities, confidence}`
  - `instructions` and criteria values may be string, object, array (rendered as JSON) or null.
  - Response `{model, answers, usage: {input_tokens, output_tokens}}`.
- `GET /v1/models` → `{models: [{name, description, release_date}]}`.
- Errors: 422 `{"detail": [{loc, msg, type}]}`; 401 when `JEV_TOKEN` is set and the bearer
  token differs; 503 when the engine fails.
- Confidence uses TypeSafe's published formulas:
  choice `(max p − 1/n)/(1 − 1/n)`; score `1 − E|i − mode| / UMAD(n)`.
- Limits: 64k-token context shared by the request; ≤32k tokens for state + longest question.
- Extension (OpenJev convention): `state.screenshot` or `state.image` holding a data URL or
  base64 PNG/JPEG is fed to the model as an image when the loaded model has a vision projector;
  otherwise 422.
- Official `typesafe-sdk` (Python) works unchanged with `TYPESAFE_BASE_URL=http://127.0.0.1:8765`.

## Engine (ADR-1)

Decision: Python engine on llama.cpp's C API via llama-cpp-python, built with Vulkan.

Options considered: (A) llama-server + HTTP shim reading top-K logprobs, least code, but the state
is prefilled once per question and the readout is lossy; (C) a custom C++ server, fastest, most code.
(B) gives the same GPU work as (C); Python overhead is ~1–2 ms per request.

Readout per request:
1. Render the model's chat template (thinking disabled) around
   `State:\n<state JSON>\n\n` + `Question: …\nOptions:\n[A] key: desc …\nAnswer with the letter of the best option only.`
   Prefix and suffix are tokenized separately and concatenated, so the prefix is shared exactly.
2. Decode the prefix once into sequence 0. If the previous request's prefix is a prefix of this
   one, decode only the new tail (agent/NPC loops with growing history).
3. `seq_cp` sequence 0 to one sequence per (question × lettering × chunk); decode all suffixes in
   as few batched `llama_decode` calls as the batch/sequence limits allow; logits only at each
   sequence's last token.
4. Softmax over the option-letter token logits divided by a fitted temperature. Letters `A–Z a–z`
   (52). More options: chunks of ≤52, then a round over chunk winners, composed as in OpenJev.
   Optional K letterings (reversed order etc.) averaged per option to cancel position bias; cheap
   because only suffixes are recomputed.
5. Remove sequences ≥1. Keep sequence 0 for reuse.
- noul: two options (true/false text) through the same readout, then a fitted logistic
  calibration (slope, bias).
- score: levels lettered in order; score = Σ i·pᵢ.
- Images: mtmd evaluates text-before-image, image, rest of prefix into sequence 0.
- Context: n_ctx 65536 by default, KV cache q8_0 + flash attention; halves until the model loads.
  Concurrent HTTP requests are serialized on one engine lock.

## Models (ADR-2)

Every local GGUF was benchmarked on JevBench public (231 items, Jev 1.13's per-item results as reference);
calibration is fitted on public data that is not JevBench. Qwen3.5-9B Q4_K_M was the most accurate. Smaller
tiers are our own models: Qwen3.5-2B and 0.8B fine-tuned (LoRA) on Jev 1.13's own probabilities from
jev-distill-corpus-v3 plus Open-Jev gold labels, through the engine's exact prompt, then merged and exported to
GGUF. JevBench public is never used for training.

Served aliases (`jev serve --model latest|vision|fast|instant|<path.gguf>`, one model per process):
`latest` 9B text-only at 64k context, `vision` 9B with its projector (16k fits), `fast` distilled 2B,
`instant` distilled 0.8B. A fast-to-big confidence cascade was tried and dropped: it scored lower than
`latest` alone and two resident models overcommitted VRAM.

## Interfaces

- `jev serve [--model] [--port 8765] [--ctx]`: on-demand server, stdlib `http.server`.
- `jev ask STATE --choice name=a,b,c --noul name="question" --score name=lo,mid,hi [--json]`.
- `godot/jev.gd`, minimal HTTPRequest helper for Godot 4 NPC decisions.

## Success criteria

- JevBench public accuracy within 3 points of Jev 1.13.
- p50 ≤100 ms warm for ≤500-token state × ≤5 questions × ≤10 options on the RX 7700S (`jev-fast`
  or `jev-latest`, whichever meets the accuracy bar; both reported).
- Zero malformed responses; official Python SDK round-trips.
- 32k-token state accepted.

## Testing

`tests/test_api.py`: starts the server on the smallest local model, checks every answer type,
the error shapes, a >52-option choice, a 255-option choice, prefix reuse and the SDK round-trip.
`bench/run.py`: JevBench public accuracy/latency per model, results kept in `bench/results/`.

## Out of scope

Billing, rate limiting, multi-model hot swapping, remote exposure (bind is 127.0.0.1; `JEV_TOKEN`
exists for anyone who changes that), streaming (Jev has none).
