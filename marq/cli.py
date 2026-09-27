# marq serve | marq ask
import argparse, json, os, sys, urllib.error, urllib.request

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(os.path.abspath(HERE))


# relative paths in models.json are from the repo root
def _path(p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_models():
    with open(os.path.join(HERE, "models.json")) as f:
        return json.load(f)


def resolve(model):
    models = load_models()
    alias = model.removeprefix("marq-").removeprefix("jev-")
    if alias in models:
        return {**models[alias], "name": f"marq-{alias}"}
    for m in models.values():  # a known path still gets its calibration
        if "path" in m and _path(m["path"]) == os.path.abspath(os.path.expanduser(model)):
            return dict(m)
    return {"path": model}


def engine_for(args):
    from .engine import Engine
    spec = resolve(args.model)
    mmproj = args.mmproj or spec.get("mmproj")
    return Engine(_path(spec["path"]), mmproj=mmproj and _path(mmproj), n_ctx=args.ctx,
                  calib=spec.get("calib"), name=spec.get("name"))


def questions(args):
    qs = {}
    for spec in args.choice or []:
        name, opts = spec.split("=", 1)
        qs[name] = {"type": "choice", "instructions": args.instructions.get(name, f"Which {name}?"),
                    "criteria": {o.strip(): None for o in opts.split(",")}}
    for spec in args.noul or []:
        name, text = spec.split("=", 1)
        qs[name] = {"type": "noul", "instructions": text}
    for spec in args.score or []:
        name, levels = spec.split("=", 1)
        qs[name] = {"type": "score", "instructions": args.instructions.get(name, f"Rate {name}."),
                    "criteria": [lv.strip() for lv in levels.split(",")]}
    if not qs:
        sys.exit("give at least one --choice, --noul or --score")
    return qs


def show(out):
    for name, a in out["answers"].items():
        if a["type"] == "noul":
            print(f"{name:14s} noul   {a['noul']:.3f}")
        else:
            top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:4]
            head = a["choice"] if a["type"] == "choice" else f"{a['score']:.2f}"
            print(f"{name:14s} {a['type']:6s} {head}  conf {a['confidence']:.2f}  " + "  ".join(f"{k}:{v:.2f}" for k, v in top))
    if "latency_ms" in out:
        print(f"({out['latency_ms']} ms, {out['usage']['input_tokens']} tokens)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="marq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the /v1/systemone server")
    s.add_argument("--model", default="latest", help="latest | vision | fast | instant | path/to/model.gguf")
    s.add_argument("--mmproj", help="vision projector .gguf (enables state.image / state.screenshot)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--ctx", type=int, default=65536, help="context tokens (halved automatically if VRAM is short)")
    a = sub.add_parser("ask", help="ask typed questions about STATE (uses a running server, else loads the model)")
    a.add_argument("state", help="state text, or @file.json / @file.txt")
    a.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    a.add_argument("--noul", action="append", metavar="NAME=question")
    a.add_argument("--score", action="append", metavar="NAME=low,mid,high")
    a.add_argument("--ask", action="append", default=[], metavar="NAME=instructions", help="instructions for a choice/score")
    a.add_argument("--url", default=os.environ.get("MARQ_URL", "http://127.0.0.1:8765"))
    a.add_argument("--token", default=os.environ.get("MARQ_TOKEN", ""))
    a.add_argument("--model", default="latest")
    a.add_argument("--mmproj")
    a.add_argument("--ctx", type=int, default=16384)
    a.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from .server import serve
        serve(engine_for(args), args.host, args.port, os.environ.get("MARQ_TOKEN", ""))
        return

    args.instructions = dict(x.split("=", 1) for x in args.ask)
    state = args.state
    if state.startswith("@"):
        with open(state[1:]) as f:
            state = f.read()
        try:
            state = json.loads(state)
        except ValueError:
            pass
    body = {"model": "marq-latest", "state": state, "questions": questions(args)}
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **({"Authorization": f"Bearer {args.token}"} if args.token else {})})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{e.code}: {e.read().decode()}")
    except OSError:  # no server running, so just load the model here
        out = engine_for(args).systemone(body)
    print(json.dumps(out, indent=2)) if args.json else show(out)


if __name__ == "__main__":
    main()
