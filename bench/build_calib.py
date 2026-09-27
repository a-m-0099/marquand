# builds bench/data/calib.jsonl (~1.2k questions) from public decision datasets, never JevBench
import json, os, random, time, urllib.error, urllib.request

HERE = os.path.dirname(__file__)
SOURCES = [
    ("ZefanCai/Open-Jev-v1.1", "community-hard-mix-v2-redistributable", "calibration", 26675, 300),
    ("SargeDev/jev-distill-corpus-v3", "default", "calibration", 13766, 500),
    ("tasksource/tasksource-jev-typed-decisions", "default", "validation", 15000, 500),
]


def rows(ds, cfg, split, total, n, seed=0):
    rnd, out = random.Random(seed), []
    for off in rnd.sample(range(0, total - 100, 100), n // 25):
        url = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"
        for wait in (0, 5, 15, 45, 90):
            time.sleep(wait)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    batch = [x["row"] for x in json.load(r)["rows"]]
                break
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
        else:
            continue
        time.sleep(1)
        out += rnd.sample(batch, min(25, len(batch)))
    return out


def convert(r):
    kind, opts, tgt = r["kind"], [str(o) for o in r["options"]], r["target"]
    state = r.get("state") if r.get("state") is not None else json.loads(r["state_json"])
    if isinstance(state, str) and len(state) > 12000 or len(opts) != len(tgt) or len(set(opts)) != len(opts):
        return None
    if kind == "choice" and 2 <= len(opts) <= 52:
        q = {"type": "choice", "instructions": r["question"], "criteria": {o: None for o in opts}}
        target = dict(zip(opts, tgt))
    elif kind == "score" and 2 <= len(opts) <= 10:
        q = {"type": "score", "instructions": r["question"], "criteria": opts}
        target = {str(i): t for i, t in enumerate(tgt)}
    elif kind == "noul" and sorted(o.lower() for o in opts) in (["no", "yes"], ["false", "true"]):
        py = sum(t for o, t in zip(opts, tgt) if o.lower() in ("yes", "true"))
        q = {"type": "noul", "instructions": r["question"]}
        target = {"yes": py, "no": 1 - py}
    else:
        return None
    return {"state": state, "question": q, "target": target}


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    n = 0
    with open(os.path.join(HERE, "data", "calib.jsonl"), "w") as f:
        for ds, cfg, split, total, want in SOURCES:
            for r in rows(ds, cfg, split, total, want):
                c = convert(r)
                if c:
                    f.write(json.dumps({"src": ds.split("/")[0], **c}) + "\n")
                    n += 1
    print(n, "calibration items")


if __name__ == "__main__":
    main()
