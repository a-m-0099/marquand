# marquand.
<img width="670" height="177" alt="image" src="https://github.com/user-attachments/assets/d08c3cfc-a8a1-4e82-9c7b-13b11cf9464b" />


### Marquand is a local drop in replacement for Typesafe AI's Jev (System One) API. 
it runs *locally on your own GPU* while still being compatible with the official 
SDK and any other things that hit `POST /v1/systemone` without touching their servers. 
#### This is an independent hobby project and is not affiliated with TypeSafe and no models here are the real Jev itself. Jev belongs to TypeSafe AI and is under *their* ownership.

Now, you may ask, *why Marquand? why not local/open jev or something?* 

Well.. 
1. OpenJEV already exists and is [something different](https://openjev.sh/).
2. Look below.

TypeSafe named Jev presumably after [*Jevons Paradox*](https://en.wikipedia.org/w/index.php?title=Jevons_paradox&oldid=1375801571&useskin=vector). 
Now, while William Jevons was the famed economist known for this, he was also a logician, and is responsible for making the logic piano in 1869, a machine that
worked out the conclusions that follow a set of premises. Allan Marquand studied Jevons' work 
and built a better one in 1881 (which was smaller, faster, and more efficient than Jevons')
and he later made the electric version, often called the first design for an electrical logic machine. So essentially, Marquand took one thing that Jevons did and made better,
which is sort of the idea of this project.

The main reason I made this is for NPCs on a personal game I'm making, which is why this project has Godot game engine compatibility (more on that in [usage](#usage), and I didn't really want to pay for Jev (no matter how cheap it is, 
it's still just a proprietary AI classifier). 
After it was working, I decided to put it publicly on GitHub in case by some miracle somebody stumbles across the repo. Maybe I'll supply updates to it later, maybe I won't.

The project was built and tested on a *Ryzen 7 7840HS* with an *RX 7700S (8 GB VRAM)* on Linux (CachyOS) using llama.cpp via Vulkan. AMD on Linux is the tested path, anything else is untested and with the current budget is not going to be planned on being tested. The 9B parameter model needs ~7.5 GB of VRAM at 64k context, the 2B needs ~4.5 GB and 0.8B a lot less. It's required to have Python 3.12, uv, cmake and glslc, and the scripts/build_llama.sh builds llama-cpp-python with Vulkan without sudo. 

In this repo, **Claude was responsible for a good bit of the model training and creating the dashboard.** Click off if you don't like that, as I would too. But it's still a solid project, 
if I don't say so myself.

## Setup

Just run `./scripts/build_llama.sh` to build.

The project has 3 models - 
- The 9B (choose between latest and vision support) is `lmstudio-community/Qwen3.5-9B-GGUF`, the `Q4_K_M` file plus `mmproj-Qwen3.5-9B-BF16.gguf` if you want images.
- "Fast" is [mine](https://huggingface.co/a-m-0099/marquand).
- "Instant" is [mine](https://huggingface.co/a-m-0099/marquand).

The models belong in `models/`. The instant model's vision projector is mmproj-Qwen3.5-0.8B-BF16.gguf from lmstudio-community/Qwen3.5-0.8B-GGUF. The paths are in `marq/models.json`, relative paths are from the repo so if you have a different setup just edit that file.
You can Use `hf download a-m-0099/marquand --local-dir models` to put the files straight into `models/`.

## Usage
#### *The part you've all been waiting for.*
*This is all tested on the `fish` shell*

Use `marq` for anything instead of marquand, which is just the project name shortened if you couldn't tell. Get the `marq` command on your shell by activating venv and then to run the server just run `marq serve`. The default model is *latest*, port 8765, localhost only. 
Try the following one-off question:
```fish
marq ask "I was charged twice and nobody answers my emails" \
  --choice team=billing,shipping,tech --noul angry="Is the customer angry?" --score urgency=low,medium,high
```

The easiest (or second-easiest to CLI, depends on who you are) way to interact with marq is through the playground/dashboard, which can be accessed by opening http://127.0.0.1:8765/

Here's the SDK working:
```python
import os; os.environ["TYPESAFE_BASE_URL"] = "http://127.0.0.1:8765"
from typesafe_sdk import TypeSafeClient, Choice, Noul
c = TypeSafeClient(api_key="local")
r = c.system_one("It arrived broken, refund me.", {
    "route": Choice(instructions="Which team?", criteria={"billing": "refunds", "engineering": "bugs"}),
    "angry": Noul(instructions="Is the customer upset?")})
```

Any key works unless MARQ_TOKEN is set on the server. Also, it only listens on 127.0.0.1 unless you pass --host. Additionally, the server keeps the GPU awake for 5 minutes after each request (--keep-warm; 0 turns it off) since on my setup my RX 7700S goes to sleep 5 seconds after its last job and waking it adds nearly a second onto the next answer. With this, a call after idle still takes around 30-60 ms (on instant model). 

### Models

The `latest ` model is 9B parameters, text only, with 64k context. It's the most accurate.

`Vision` is the same 9B model except it has image support, but only 16k context fits with the projector loaded.

`fast ` is my distilled 2B parameter model with vision; this one's pretty good for NPCs and loops and it has a lot of 9B's accuracy with much lower latency.

`instant` is my distilled 0.8B model with vision and is for when near zero latency is top priority. 

This is all a one model per server process so you pick with `marq serve --model [name]`. 

Marquand also accepts the jev-latest model name since that's what the official SDK sends by default.

**To add images**: put a data URL or base64 PNG/JPEG/GIF/WEBP in `state.image` or `state.screenshot` and use vision, fast, or instant. In the playground just upload the image.  

----------------
### Godot

`godot/marq.gd` is an autoload called Marq with a single coroutine. Example:
```gdscript
var a := await Marq.decide({"hunger": needs.hunger_level, "energy": needs.energy_level}, {
    "next": {"type": "choice", "instructions": "What should this villager do next?",
             "criteria": {"eat": "go eat", "rest": "go to bed", "wander": "walk around", "idle": "stay"}}})
planner.append(a.get("next", {}).get("choice", "idle"))
```

`godot/test` is a headless check.

## Comparisons

### Marquand leads
| | Marquand | Jev |
|---|---|---|
| Where it runs | your machine, offline | TypeSafe's cloud only |
| Cost | free (electricity) | $0.042 per 1M input tokens |
| Rate limits | none (one request at a time per process) | 1,200 req/min, 250k tokens/s (early access) |
| Access | clone and run | new sign-ups were paused on 2026-09-22 |
| Privacy | state never leaves the laptop | state is sent to TypeSafe |
| Images | yes: `state.image` / `state.screenshot` on `vision`, `fast`, `instant` | text only ("no images, audio or video yet") |
| Models | 4 tiers, and you can swap in any GGUF | one hosted model |
| Weights | open; retrain with `train/distill.py` | closed |
| Latency floor | 69 ms for a 5-question call (`instant`, question dependent), no network | ~70-500 ms per call, including the network |
| Extras | CLI, Godot autoload, prefix reuse for realtime loops, thermal and VRAM guards | JS/Python SDKs, Vercel AI SDK integration |

### Feature Parity
| | Both |
|---|---|
| API | `POST /v1/systemone` with `state` + typed `questions`; `GET /v1/models`; the official Python SDK works against Marquand unchanged (tested) |
| Question types | `noul` (0-1), `choice` (up to 255 options), `score` (2-10 levels, expected value + legend) |
| Confidence | same formulas (TypeSafe's published ones) |
| Output | typed answers only; malformed output is structurally impossible |
| Fan-out | many questions against one state in a single call; the state is processed once |
| Context | 64k shared, 32k state + question (Marquand: 64k on `latest`/`fast`/`instant`, 16k on `vision`) |
| Errors | 401 / 422 (same `detail[]` shape) / 503 |

### Jev leads
| | Marquand | Jev 1.13 |
|---|---|---|
| JevBench public (231) | 188 `latest`, 163 `fast`, 147 `instant` | 200 |
| Hard tier (111) | 74 / 52 / 44 | 81 |
| Standard tier (72) | 66 / 63 / 55 | 71 |
| Per-item latency, best model | 247 ms p50 (9B) | ~100 ms typical |
| Calibration | fitted temperature + position bias (ECE 0.045 on `latest`) | trained for calibration (RLCD) |
| Throughput | single GPU, requests serialized | hosted, parallel |
| Hardware needed | ~7.5 GB VRAM for the 9B (4.5 GB `fast`) | none (cloud) |

### Marquand vs. other open Jev alternatives (231 item bench) 
Marquand `latest` 188, SemIf-4B 187, reflex-4B 183, Open-Jev-9B 179, decider-2B 164.

## How it works
##### *this parts pretty cool imo*

The state gets prefilled once and then every question branches off of it, all of which is then decoded in one batch, with the answer being the softmax
over the option letters at the answer spot (*+position bias and temp*),the confidence uses TypeSafe's published formulas, and the realtime loops just reuse the cache so just the changed part of the state is re-read. `docs/design.md` has the full write-up and the API contract, so take a look at that.

## Training

If you want to make your own fast model, `train/distill.py`  fine-tunes `Qwen3.5` on Jev's own probabilities (`jev-distill-corpus-v3`) along with Open-Jev labels. `train/export.py` merges it and converts it into a GGUF. It requires a separate PyTorch ROCm venv; the one I used was:
```fish
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train --index-url https://download.pytorch.org/whl/rocm7.14 torch
uv pip install --python .venv-train transformers peft accelerate safetensors numpy pyarrow flash-linear-attention bitsandbytes
```

It also needs the base weights for Qwen3.5 in `train/base`, the datasets in `train/data`, and the llama.cpp `convert_hf_to_gguf.py` in `train/llamacpp`
It took me about 1.5 hours for the 2B model and 2 hours for the 0.8B, though those numbers may have been a bit botched - more on that below.

#### Laptop Safety Note

Unfortunately, while I was training these models on my laptop (Framework 16 Arch Linux btw), I ran into some quite high temperatures ~100 degrees C since my training was (obviously) rather poorly planned and did not account for temps and RAM spills / OOM. Training and benchmarking pauses at 85 degrees C and resume at 75 degrees C (`MARQ_MAX_TEMP` / `MARQ_RESUME_TEMP`). The engine won't let a model run out of VRAM and spill into RAM - this OOM-ed me. The `scripts/memguard.sh` script kills a job if free RAM is below 2.5 gigs. Also, `ROCPROFILER_REGISTER_ENABLED`=0 because otherwise it keeps a CPU core at 100% the whole time and you don't want a freshly fried CPU, do you?

## Testing + Benchmarks
`.venv/bin/pytest -q` runs 28 tests; engine, vision, HTTP, API, official SDK round-trip, CLI, guards. They skip if the test model isn't there
(`MARQ_TEST_MODEL`). For the benchmarks, JevBench gets cloned into `bench/jevbench` and `bench/build_calib.py` rebuilds calibration data before `bench/calib.py` can refit anything;
```fish
git clone --depth 1 https://github.com/fstandhartinger/jevbench bench/jevbench
.venv/bin/python bench/run.py latest --calibrated --latency
```

## Limitations

Marquand is still behind 12 points from Jev overall and further behind on hard tier. It's slower than their hosted API unless you use fast or instant. It only runs a single model per process. And finally, it's only been tested on AMD + Linux, CachyOS to be specific.

## *Credits and Licenses*

Code is licensed under MIT. Credit to llama.cpp and llama-cpp-python, Qwen 3.5 which is under the Apache 2.0 license, JevBench, OpenJev, and the jev-distill-corpus-v3, Open-Jev-v1.1, and tasksource datasets.
