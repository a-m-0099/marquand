# fits a model's calibration (temperature, letter-position bias, yes/no scaling) on bench/data/calib.jsonl,
# never JevBench, and writes it into marq/models.json with --alias
#   .venv/bin/python bench/calib.py MODEL.gguf --alias latest
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from marq.engine import Engine  # noqa: E402
from marq.thermal import guard  # noqa: E402

HERE = os.path.dirname(__file__)
NMAX = 10


def collect(E, rows):
    out = []
    for r in rows:
        guard()
        E.trace = []
        E.systemone({"state": r["state"], "questions": {"q": r["question"]}})
        q = r["question"]
        keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else ["yes", "no"]
        tgt = np.array([r["target"].get(k, 0.0) for k in keys], float)
        if not E.trace or tgt.sum() <= 0:
            continue
        orders, zs = E.trace[0]
        out.append({"type": q["type"], "n": len(keys), "tgt": tgt / tgt.sum(), "orders": orders, "z": [np.array(z) for z in zs]})
    E.trace = None
    return out


def probs(item, T, bias, perms):
    n, acc = item["n"], np.zeros(item["n"])
    b = bias.get(n, np.zeros(n))
    use = list(zip(item["orders"], item["z"]))[:perms]
    for o, z in use:
        x = (z - b) / T
        e = np.exp(x - x.max())
        acc[o] += e / e.sum() / len(use)
    return acc


def nll(items, T, bias, perms, noul=(1.0, 0.0)):
    tot, ok = 0.0, 0
    for it in items:
        p = probs(it, T, bias, perms)
        if it["type"] == "noul":
            py = np.clip(p[0], 1e-6, 1 - 1e-6)
            y = 1 / (1 + np.exp(-(np.log(py / (1 - py)) / noul[0] + noul[1])))
            p = np.array([y, 1 - y])
        p = np.clip(p, 1e-9, 1)
        tot -= float((it["tgt"] * np.log(p)).sum())
        ok += int(np.argmax(p) == np.argmax(it["tgt"]))
    return tot / len(items), ok / len(items)


def fit(train, perms):
    T, bias, noul = 1.0, {}, (1.0, 0.0)
    for _ in range(3):
        T = min(np.geomspace(0.3, 5, 40), key=lambda t: nll([i for i in train if i["type"] != "noul"], t, bias, perms)[0])
        for n in range(2, NMAX + 1):
            sub = [i for i in train if i["n"] == n and i["type"] != "noul"] + ([i for i in train if i["type"] == "noul"] if n == 2 else [])
            if len(sub) < 20:
                continue
            b = bias.get(n, np.zeros(n)).copy()
            for _ in range(2):
                for k in range(1, n):
                    grid = b[k] + np.linspace(-3, 3, 25)
                    best = min(grid, key=lambda v: nll(sub, T, {**bias, n: np.r_[b[:k], v, b[k + 1:]]}, perms)[0])
                    b[k] = best
            bias[n] = b
        nt = [i for i in train if i["type"] == "noul"]
        if nt:
            noul = min(((t, c) for t in np.geomspace(0.3, 5, 25) for c in np.linspace(-2, 2, 21)),
                       key=lambda tc: nll(nt, T, bias, perms, tc)[0])
    return T, bias, noul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--alias")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ctx", type=int, default=16384)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(HERE, "data", "calib.jsonl"))][:a.limit]
    E = Engine(os.path.expanduser(a.model), n_ctx=a.ctx, calib={"perms": 2})
    items = collect(E, rows)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(items))
    cut = int(len(items) * 0.7)
    train, test = [items[i] for i in idx[:cut]], [items[i] for i in idx[cut:]]
    report, best = {}, None
    for perms in (1, 2):
        base = nll(test, 1.0, {}, perms)
        T, bias, noul = fit(train, perms)
        after = nll(test, T, bias, perms, noul)
        report[perms] = {"raw_nll_acc": base, "fit_nll_acc": after, "T": float(T), "noul": [float(x) for x in noul]}
        print(f"perms={perms}: held-out NLL/acc raw {base[0]:.3f}/{base[1]:.3f} -> fitted {after[0]:.3f}/{after[1]:.3f}  T={T:.2f} noul={noul}", flush=True)
        cand = {"T": round(float(T), 3), "noul_T": round(float(noul[0]), 3), "noul_b": round(float(noul[1]), 3), "perms": perms,
                "bias": {str(n): [round(float(v), 3) for v in b] for n, b in bias.items()}}
        # one lettering is half the tokens, so two only wins if it's clearly better (1.5+ points)
        if best is None or after[1] > best[1] + 0.015:
            best = (cand, after[1])
    print("chosen", json.dumps(best[0]))
    if a.alias:
        path = os.path.join(HERE, "..", "marq", "models.json")
        models = json.load(open(path))
        models[a.alias]["calib"] = best[0]
        json.dump(models, open(path, "w"), indent=2)
    json.dump(report, open(os.path.join(HERE, "results", f"calib-{E.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
