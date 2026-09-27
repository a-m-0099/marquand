"""JevBench public (231 items) + latency microbench for one local model.

  .venv/bin/python bench/run.py MODEL.gguf [--perms 2] [--T 1.0] [--latency] [--limit N]
Writes bench/results/<model>[-tag].json and prints accuracy per tier next to Jev 1.13.
"""
import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jev.engine import Engine  # noqa: E402
from jev.thermal import guard  # noqa: E402

HERE = os.path.dirname(__file__)
PUB = os.path.join(HERE, "jevbench", "datasets", "public")
TIERS = {"original": "standard", "easy": "easy", "hard": "hard"}
JEV = {"easy": 48, "standard": 71, "hard": 81, "total": 200}  # Jev 1.13.0 public items correct (JevBench v1.3 per-task)


def items(limit=None):
    out = []
    for f, tier in TIERS.items():
        for line in open(os.path.join(PUB, f + ".jsonl")):
            r = json.loads(line)
            r["tier"] = tier
            out.append(r)
    return out[:limit] if limit else out


def probs_of(ans):
    if ans["type"] == "noul":
        return {"yes": ans["noul"], "no": 1 - ans["noul"]}
    return ans["probabilities"]


def argmax(p):
    return min(p, key=lambda k: (-p[k], k))  # JevBench tie rule: smallest label


def run(engine, rows):
    res = []
    for r in rows:
        guard()
        t0 = time.perf_counter()
        out = engine.systemone({"model": "jev-latest", "state": r["state"], "questions": {"q": r["question"]}})
        ms = (time.perf_counter() - t0) * 1000
        p = probs_of(out["answers"]["q"])
        pred = argmax(p)
        res.append({"id": r["id"], "tier": r["tier"], "family": r.get("family"), "pred": pred, "gold": str(r["expected"]),
                    "ok": pred == str(r["expected"]), "p": p, "ms": ms, "tokens": out["usage"]["input_tokens"]})
    return res


def summarize(res):
    s = {}
    for tier in ["easy", "standard", "hard"]:
        rr = [x for x in res if x["tier"] == tier]
        s[tier] = sum(x["ok"] for x in rr)
    s["total"] = sum(x["ok"] for x in res)
    s["n"] = len(res)
    s["brier"] = statistics.mean(sum((v - (k == x["gold"])) ** 2 for k, v in x["p"].items()) for x in res)
    bins = [[] for _ in range(10)]
    for x in res:
        c = max(x["p"].values())
        bins[min(int(c * 10), 9)].append((c, x["ok"]))
    s["ece"] = sum(len(b) / len(res) * abs(statistics.mean(c for c, _ in b) - statistics.mean(o for _, o in b)) for b in bins if b)
    ms = sorted(x["ms"] for x in res)
    s["p50_ms"], s["p95_ms"] = ms[len(ms) // 2], ms[int(len(ms) * 0.95)]
    return s


def latency(engine, n=40):
    """Realtime shape: ~300-token state x 5 questions x 6 options, state changes every call."""
    base = ("NPC Mara, blacksmith. Hunger 72/100, energy 35/100, gold 14. Time 18:40, raining. "
            "Nearby: tavern (open, 40 m), forge (her own, 5 m), market (closing, 120 m), home (200 m). "
            "Recent: finished two swords, argued with guard Tomas about taxes, customer waiting at forge. ") * 3
    qs = {
        "next": {"type": "choice", "instructions": "What should Mara do next?",
                 "criteria": {"eat": "go to the tavern to eat", "work": "serve the waiting customer", "shop": "buy iron at the market",
                              "rest": "go home and sleep", "talk": "talk to guard Tomas", "idle": "stay put"}},
        "mood": {"type": "choice", "instructions": "Mara's mood?", "criteria": {k: None for k in ["happy", "tired", "angry", "hungry", "calm", "anxious"]}},
        "hungry": {"type": "noul", "instructions": "Is Mara hungry enough that eating should come first?"},
        "urgency": {"type": "score", "instructions": "How urgent is the waiting customer?", "criteria": ["not", "a bit", "somewhat", "quite", "very", "critical"]},
        "danger": {"type": "noul", "instructions": "Is Mara in physical danger?"},
    }
    engine.systemone({"state": base, "questions": qs})  # warm
    ms = []
    for i in range(n):
        guard()
        t0 = time.perf_counter()
        engine.systemone({"state": base + f"Tick {i}.", "questions": qs})
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return {"p50_ms": ms[n // 2], "p95_ms": ms[int(n * 0.95)], "state_chars": len(base)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mmproj")
    ap.add_argument("--perms", type=int, default=2)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--calibrated", action="store_true", help="MODEL is a jev/models.json alias; use its calibration")
    a = ap.parse_args()
    if a.calibrated:  # alias from jev/models.json, with its calibration
        from jev.cli import engine_for
        e = engine_for(argparse.Namespace(model=a.model, mmproj=a.mmproj, ctx=a.ctx))
    else:
        e = Engine(os.path.expanduser(a.model), mmproj=a.mmproj, n_ctx=a.ctx, calib={"perms": a.perms, "T": a.T})
    name = e.name + (f"-{a.tag}" if a.tag else "")
    out = {"model": name, "calib": e.calib}
    if a.latency:
        out["latency"] = latency(e)
        print(name, "latency", out["latency"], flush=True)
    res = run(e, items(a.limit))
    out["summary"], out["rows"] = summarize(res), res
    s = out["summary"]
    print(f"{name}: {s['total']}/{s['n']} (Jev {JEV['total']}/231)  easy {s['easy']}/48 std {s['standard']}/72 "
          f"hard {s['hard']}/111 (Jev hard {JEV['hard']})  brier {s['brier']:.3f} ece {s['ece']:.3f}  "
          f"p50 {s['p50_ms']:.0f}ms p95 {s['p95_ms']:.0f}ms", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", name + ".json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
