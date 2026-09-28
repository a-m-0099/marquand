import os, pytest
from marq.engine import Engine, BadRequest, choice_confidence, score_confidence

M = os.environ.get("MARQ_TEST_MODEL", os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"))
if not os.path.exists(M):
    pytest.skip(f"test model {M} not found; set MARQ_TEST_MODEL", allow_module_level=True)
E = Engine(M, n_ctx=8192)


def ask(state, **qs):
    return E.systemone({"model": "marq-latest", "state": state, "questions": qs})


def test_confidence_formulas():
    assert choice_confidence([1.0]) == 1.0
    assert abs(choice_confidence([0.5, 0.5])) < 1e-9
    assert score_confidence([0, 0, 1]) == 1.0


def test_all_types_and_shape():
    r = ask("Customer: I was charged twice for my order and nobody replies!",
            route={"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "payments, charges, refunds", "shipping": "delivery and tracking", "tech": "app bugs"}},
            angry={"type": "noul", "instructions": "Is the customer angry?"},
            urgency={"type": "score", "instructions": "How urgent?",
                     "criteria": ["can wait", "this week", "today", "now"]})
    a = r["answers"]
    assert a["route"]["type"] == "choice" and a["route"]["choice"] == "billing"
    assert abs(sum(a["route"]["probabilities"].values()) - 1) < 1e-3
    assert a["angry"]["type"] == "noul" and a["angry"]["noul"] > 0.5
    assert set(a["urgency"]["probabilities"]) == {"0", "1", "2", "3"}
    assert a["urgency"]["legend"]["0"] == "can wait" and 0 <= a["urgency"]["score"] <= 3
    assert r["usage"]["input_tokens"] > 0 and r["model"]


def test_one_option_and_json_values():
    r = ask({"msg": "hi"}, only={"type": "choice", "instructions": {"task": "pick"},
                                 "criteria": {"x": {"desc": ["a", 1]}}})
    assert r["answers"]["only"] == {"type": "choice", "choice": "x", "probabilities": {"x": 1.0}, "confidence": 1.0}


def test_many_options():
    crit = {f"opt{i}": f"the number {i}" for i in range(255)}
    r = ask("The number is 137.", n={"type": "choice", "instructions": "Which option matches?", "criteria": crit})
    p = r["answers"]["n"]["probabilities"]
    assert len(p) == 255 and abs(sum(p.values()) - 1) < 1e-2


def test_validation():
    with pytest.raises(BadRequest):
        ask("x", s={"type": "score", "criteria": ["only one"]})
    with pytest.raises(BadRequest):
        ask("x", c={"type": "choice", "criteria": {}})
    with pytest.raises(BadRequest):
        E.systemone({"state": "x", "questions": {}})
    with pytest.raises(BadRequest):
        ask("x", c={"type": "choice", "criteria": {f"o{i}": None for i in range(256)}})


def test_prefix_reuse_same_answer():
    q = {"t": {"type": "noul", "instructions": "Is the sky mentioned?"}}
    a = E.systemone({"state": "The sky is blue.", "questions": q})
    b = E.systemone({"state": "The sky is blue.", "questions": q})
    c = E.systemone({"state": "The sky is blue. It is noon.", "questions": q})
    assert abs(a["answers"]["t"]["noul"] - b["answers"]["t"]["noul"]) < 1e-3
    assert c["answers"]["t"]["noul"] > 0.5


def test_checkpoint_restore_matches_fresh():
    base = "Log line about the forge and the weather. " * 60  # long enough for a few checkpoints
    q = {"t": {"type": "choice", "instructions": "What was the last tick?", "criteria": {"one": None, "two": None}}}
    E.systemone({"state": base + "Tick one.", "questions": q})
    warm = E.systemone({"state": base + "Tick two.", "questions": q})
    reused = E.prefilled
    E.cached = []  # force a full prefill
    cold = E.systemone({"state": base + "Tick two.", "questions": q})
    assert reused < E.prefilled / 2  # it restored a checkpoint instead of redoing everything
    assert warm["answers"]["t"]["probabilities"] == pytest.approx(cold["answers"]["t"]["probabilities"], abs=2e-3)


# plain color PNG with just the stdlib
def _png(rgb, size=64):
    import struct, zlib
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_image_state():
    import base64
    mm = os.environ.get("MARQ_TEST_MMPROJ", os.path.expanduser(
        "~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/mmproj-Qwen3.5-0.8B-BF16.gguf"))
    if not os.path.exists(mm):
        pytest.skip(f"vision projector {mm} not found; set MARQ_TEST_MMPROJ")
    V = Engine(M, mmproj=mm, n_ctx=8192)
    q = {"color": {"type": "choice", "instructions": "What color is the image?",
                   "criteria": {"red": None, "green": None, "blue": None}}}
    for rgb, want in [((230, 20, 20), "red"), ((20, 200, 40), "green"), ((20, 40, 230), "blue")]:
        url = "data:image/png;base64," + base64.b64encode(_png(rgb)).decode()
        r = V.systemone({"state": {"image": url, "note": "a test swatch"}, "questions": q})
        assert r["answers"]["color"]["choice"] == want
    with pytest.raises(BadRequest):  # a text-only model says no to images
        E.systemone({"state": {"image": url}, "questions": q})


def test_failed_prefill_does_not_poison_cache():
    q = {"t": {"type": "choice", "instructions": "Ball color?", "criteria": {"red": None, "blue": None}}}
    long = "The ball is red. " + "Filler sentence about nothing. " * 80
    fresh = E.systemone({"state": long, "questions": q})["answers"]
    orig, calls = E._decode, []

    def flaky(items):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated GPU failure")
        return orig(items)
    E._decode = flaky
    try:
        with pytest.raises(RuntimeError):
            E.systemone({"state": long + " Extra tail.", "questions": q})
    finally:
        E._decode = orig
    again = E.systemone({"state": long, "questions": q})["answers"]
    assert again["t"]["probabilities"] == pytest.approx(fresh["t"]["probabilities"], abs=2e-3)


def test_state_over_loaded_context_is_422():
    with pytest.raises(BadRequest):
        ask("word " * 12000, q={"type": "noul", "instructions": "Any words?"})  # bigger than the 8192 test context but under the 32k limit


def test_long_text_image_field_is_just_text():
    r = ask({"image": "a painting of a harbor at dusk " * 100}, q={"type": "noul", "instructions": "Is a harbor described?"})
    assert r["answers"]["q"]["type"] == "noul"


def test_ping_leaves_answers_alone():
    body = {"state": "The ball is red and the box is blue.", "questions": {
        "c": {"type": "choice", "instructions": "Ball color?", "criteria": {"red": None, "blue": None}}}}
    a = E.systemone(body)["answers"]
    E.ping()
    b = E.systemone(body)["answers"]  # same prompt, so this goes through the cache reuse path
    assert a == b
