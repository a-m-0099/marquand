# trains a small Qwen3.5 to match Jev's own probabilities, through the exact prompt the engine uses
#   ../scripts/memguard.sh ../.venv-train/bin/python distill.py --base base/Qwen3.5-2B --out runs/fast2b --n 16000
# add --load-4bit for the 4B, and rerun the same command to pick up from the last checkpoint
import argparse, json, math, os, random, sys, time

os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")  # without this a CPU core sits at 100% the whole run
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")  # the RX 7700S runs the gfx1100 kernels
import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from marq import prompt as P  # noqa: E402
from marq.thermal import guard, hottest  # noqa: E402
from bench.build_calib import convert  # noqa: E402

HEAD = "<|im_start|>user\n"  # has to match what the engine sends
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


def load_rows(n, seed):
    rnd = random.Random(seed)
    want = {"yuri_v3": int(n * 0.7), "openjev_v2": int(n * 0.15)}
    res, seen = {k: [] for k in want}, {k: 0 for k in want}
    for line in open(os.path.join(HERE, "data", "distill_train.jsonl")):  # sampled on the fly so 656k rows never sit in RAM
        r = json.loads(line)
        k = r["source"]
        if k not in want:
            continue
        seen[k] += 1
        if len(res[k]) < want[k]:
            res[k].append(r)
        elif (j := rnd.randrange(seen[k])) < want[k]:
            res[k][j] = r
    import pyarrow.parquet as pq
    hard = pq.read_table(os.path.join(HERE, "data", "openjev_train.parquet")).to_pylist()
    pick = res["yuri_v3"] + res["openjev_v2"] + rnd.sample(hard, int(n * 0.15))
    rows = [c for c in (convert(r) for r in pick) if c]
    rnd.shuffle(rows)
    return rows


def eval_rows(k, seed=1):
    rows = [json.loads(line) for line in open(os.path.join(HERE, "data", "distill_test30k.jsonl"))]
    rows = [r for r in rows if r["source"] == "yuri_v3"]
    return [c for c in (convert(r) for r in random.Random(seed).sample(rows, k)) if c]


class Encoder:
    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len
        self.head = tok.encode(HEAD, add_special_tokens=False)
        self.tail = tok.encode(TAIL, add_special_tokens=False)
        self.letters = [tok.encode(c, add_special_tokens=False)[0] for c in P.LETTERS]

    def __call__(self, row, rnd):
        instr, opts = P.options(row["question"])
        keys = [k for k, _ in opts] if row["question"]["type"] != "noul" else ["yes", "no"]
        tgt = [row["target"].get(k, 0.0) for k in keys]
        if len(opts) > len(P.LETTERS) or sum(tgt) <= 0:
            return None
        order = list(range(len(opts)))
        rnd.shuffle(order)  # new letter order every time so no letter learns an answer
        ids = (self.head + self.tok.encode(P.prefix(P.text(row["state"])), add_special_tokens=False)
               + self.tok.encode(P.question(instr, [opts[i] for i in order]), add_special_tokens=False) + self.tail)
        if len(ids) > self.max_len:
            return None
        s = sum(tgt)
        return ids, [tgt[i] / s for i in order]


def readout(model, ids, n, letters, device):
    logits = model(input_ids=torch.tensor([ids], device=device), logits_to_keep=1).logits[0, -1]
    return torch.log_softmax(logits[letters[:n]].float(), -1)


@torch.no_grad()
def evaluate(model, enc, rows, device):
    model.eval()
    rnd, agree, kl, n = random.Random(7), 0, 0.0, 0
    for r in rows:
        ex = enc(r, rnd)
        if not ex:
            continue
        ids, t = ex
        lp = readout(model, ids, len(t), enc.letters, device)
        tt = torch.tensor(t, device=device)
        agree += int(lp.argmax().item() == int(tt.argmax().item()))
        kl += float((tt * (torch.log(tt.clamp_min(1e-9)) - lp)).sum())
        n += 1
    model.train()
    return agree / max(n, 1), kl / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-4bit", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = a.device
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.base)
    enc = Encoder(tok, a.max_len)
    if a.load_4bit:  # 4-bit base so the 4B fits in 8 GB
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, quantization_config=q, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    progress = os.path.join(a.out, "progress.json")
    done = json.load(open(progress))["seen"] if os.path.exists(progress) else 0
    if done:
        model = PeftModel.from_pretrained(model, a.out, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, target_modules=TARGETS))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    rows, ev = load_rows(a.n, 0), eval_rows(a.eval_n)
    print(f"{len(rows)} train rows, {len(ev)} eval rows", flush=True)
    base_eval = evaluate(model, enc, ev, device)
    print(json.dumps({"step": 0, "agree": base_eval[0], "kl": base_eval[1]}), flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = len(rows) // a.accum
    s0 = done // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + s0 + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s + s0, total) / total))))
    rnd, t0, run_loss, seen, step = random.Random(0), time.time(), 0.0, 0, 0
    for i, r in enumerate(rows):
        ex = enc(r, rnd)
        if not ex:
            continue
        if seen < done:  # already done before the resume
            seen += 1
            step += seen % a.accum == 0
            continue
        ids, t = ex
        with torch.autocast(device, dtype=torch.bfloat16):
            lp = readout(model, ids, len(t), enc.letters, device)
        loss = -(torch.tensor(t, device=device) * lp).sum()
        (loss / a.accum).backward()
        run_loss += loss.item()
        seen += 1
        if seen % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            guard()
            if step % 20 == 0:
                print(json.dumps({"step": step, "of": total, "loss": round(run_loss / (20 * a.accum), 4),
                                  "ex_s": round((seen - done) / (time.time() - t0), 2), "temp": hottest()}), flush=True)
                run_loss = 0.0
        if seen % a.eval_every == 0:
            ag, kl = evaluate(model, enc, ev, device)
            print(json.dumps({"step": step, "agree": round(ag, 4), "kl": round(kl, 4)}), flush=True)
        if seen % 1600 == 0:
            model.save_pretrained(a.out)
            json.dump({"seen": seen}, open(progress, "w"))
    ag, kl = evaluate(model, enc, ev, device)
    print(json.dumps({"final": True, "agree": round(ag, 4), "kl": round(kl, 4), "base_agree": base_eval[0]}), flush=True)
    model.save_pretrained(a.out)
    json.dump({"seen": seen, "final": True}, open(progress, "w"))


if __name__ == "__main__":
    main()
