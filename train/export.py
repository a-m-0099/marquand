"""Merge a LoRA run into the original Qwen3.5 checkpoint one tensor at a time and convert to GGUF.

  ../.venv-train/bin/python export.py --base base/Qwen3.5-4B --adapter runs/mid4b --out ../models/jev-mid-4b --outtype q8_0

Streams: never more than ~1 GB of weights in RAM, so a 4B model exports on a 14 GB laptop. Vision and MTP tensors are
copied untouched, so the base model's mmproj still applies.
"""
import argparse, json, os, shutil, subprocess, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
FLUSH_BYTES = 1 << 30


def lora_deltas(adapter):
    """{checkpoint tensor name: (A, B)} and the LoRA scale, from a PEFT adapter directory."""
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    f = safe_open(os.path.join(adapter, "adapter_model.safetensors"), "pt")
    out = {}
    for k in f.keys():
        if k.endswith(".lora_A.weight"):
            mod = k[: -len(".lora_A.weight")].removeprefix("base_model.model.")  # model.layers.3.mlp.up_proj
            name = mod.replace("model.", "model.language_model.", 1) + ".weight"
            out[name] = (f.get_tensor(k), f.get_tensor(k.replace("lora_A", "lora_B")))
    return out, cfg["lora_alpha"] / cfg["r"]


def merge(base, adapter, out_dir):
    deltas, scale = lora_deltas(adapter)
    idx_path = os.path.join(base, "model.safetensors.index.json")
    shards = sorted(set(json.load(open(idx_path))["weight_map"].values())) if os.path.exists(idx_path) else \
        [f for f in os.listdir(base) if f.endswith(".safetensors")]
    os.makedirs(out_dir, exist_ok=True)
    weight_map, buf, size, n_out, used = {}, {}, 0, 0, set()

    def flush():
        nonlocal buf, size, n_out
        if buf:
            n_out += 1
            name = f"model-{n_out:05d}.safetensors"
            save_file(buf, os.path.join(out_dir, name), metadata={"format": "pt"})
            weight_map.update({k: name for k in buf})
            buf, size = {}, 0

    for shard in shards:
        with safe_open(os.path.join(base, shard), "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k in deltas:
                    a, b = deltas[k]
                    t = (t.float() + scale * (b.float() @ a.float())).to(t.dtype)
                    used.add(k)
                buf[k] = t.contiguous()
                size += t.numel() * t.element_size()
                if size >= FLUSH_BYTES:
                    flush()
    flush()
    missing = set(deltas) - used
    if missing:
        raise KeyError(f"adapter tensors with no checkpoint match: {sorted(missing)[:3]}")
    json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    for f in os.listdir(base):
        if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
            shutil.copy(os.path.join(base, f), out_dir)
    return len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", default="q8_0")
    a = ap.parse_args()
    hf = a.out + "-hf"
    print(f"{merge(a.base, a.adapter, hf)} tensors merged", flush=True)
    out = a.out + f"-{a.outtype.upper()}.gguf"
    subprocess.run([sys.executable, os.path.join(HERE, "llamacpp", "convert_hf_to_gguf.py"), hf, "--outtype", a.outtype, "--outfile", out], check=True)
    print(out)


if __name__ == "__main__":
    main()
