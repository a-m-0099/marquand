import json, os, subprocess, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import pytest

PORT = 8799
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("MARQ_TEST_MODEL", os.path.expanduser(
    "~/.lmstudio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"))


if not os.path.exists(MODEL):
    pytest.skip(f"test model {MODEL} not found; set MARQ_TEST_MODEL", allow_module_level=True)


def call(path, body=None, token=None):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(body, token=None):
    return call("/v1/systemone", body, token)


@pytest.fixture(scope="module", autouse=True)
def server():
    p = subprocess.Popen([sys.executable, "-m", "marq.cli", "serve", "--model", MODEL, "--port", str(PORT), "--ctx", "8192"],
                         env={**os.environ, "MARQ_TOKEN": "sekret"}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    for _ in range(600):
        if p.poll() is not None:  # it died (usually not enough free VRAM), so say why instead of waiting it out
            pytest.fail("test server exited:\n" + p.stderr.read()[-800:])
        try:
            urllib.request.urlopen(URL + "/health", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    yield p
    p.terminate()
    p.wait()


def test_auth():
    assert post({"state": "x", "questions": {"q": {"type": "noul"}}})[0] == 401
    assert post({"state": "x", "questions": {"q": {"type": "noul"}}}, "sekret")[0] == 200


def test_models():
    code, body = call("/v1/models", token="sekret")
    assert code == 200 and body["models"][0]["name"] == "jev-latest"  # the SDK asks for this by default


def test_422_shape():
    code, body = post({"state": "x"}, "sekret")
    assert code == 422 and body["detail"][0]["loc"][:2] == ["body", "questions"]


def test_oversize_state_422_and_alive():
    code, body = post({"state": "word " * 40000, "questions": {"q": {"type": "noul"}}}, "sekret")
    assert code == 422 and body["detail"][0]["loc"] == ["body", "state"]
    assert call("/v1/models", token="sekret")[0] == 200


def test_concurrent_isolation():
    colors = ["red", "green", "blue", "yellow"]
    states = [f"The secret color is {c}." for c in colors * 2]
    q = {"c": {"type": "choice", "instructions": "What is the secret color?", "criteria": {k: None for k in colors}}}
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(lambda s: post({"state": s, "questions": q}, "sekret")[1]["answers"]["c"]["choice"], states))
    assert outs == colors * 2


def test_sdk_roundtrip():
    sdk = pytest.importorskip("typesafe_sdk")
    c = sdk.TypeSafeClient(api_key="sekret", base_url=URL)
    r = c.system_one("It arrived broken, refund me.", {
        "route": sdk.Choice(instructions="Which team?", criteria={"billing": "refunds and charges", "engineering": "software bugs"}),
        "angry": sdk.Noul(instructions="Is the customer upset?")})
    assert r.answers["route"].choice == "billing"


def test_cli_ask():
    out = subprocess.run([sys.executable, "-m", "marq.cli", "ask", "--url", URL, "--token", "sekret", "--json",
                          "I want to cancel my order", "--choice", "intent=cancel,track,refund", "--noul", "polite=Is it polite?",
                          "--score", "urgency=low,medium,high"], capture_output=True, text=True, check=True).stdout
    a = json.loads(out)["answers"]
    assert a["intent"]["choice"] == "cancel" and "noul" in a["polite"] and "score" in a["urgency"]


def test_playground_page():
    with urllib.request.urlopen(URL + "/", timeout=10) as r:
        html = r.read().decode()
    assert r.status == 200 and r.headers["Content-Type"].startswith("text/html") and "/v1/systemone" in html
