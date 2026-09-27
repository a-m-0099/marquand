"""Jev-compatible typed decisions on a local GGUF model.

One request = one prefill of the shared state into sequence 0, then every question (x lettering x chunk)
gets its own sequence copied from 0 and only its suffix is decoded, all in as few batches as possible.
The answer is the softmax over the option-letter logits at the first answer position.
"""
import base64, ctypes, glob, hashlib, json, math, os, time

# RX 7700S is Vulkan device 1 here (0 = 780M iGPU); keeps layers off the iGPU. Override with the env var.
os.environ.setdefault("GGML_VK_VISIBLE_DEVICES", "1")
import llama_cpp as L  # noqa: E402

from .prompt import LETTERS, options, prefix, question, text as _text  # noqa: E402
MAX_OPTIONS, MIN_LEVELS, MAX_LEVELS, MAX_TOKENS = 255, 2, 10, 32768
DEFAULT_CALIB = {"T": 1.0, "noul_T": 1.0, "noul_b": 0.0, "perms": 2, "bias": {}}
IMAGE_KEYS = ("screenshot", "image")
CKPT_EVERY, N_CKPT = 128, 4  # prefill checkpoints for hybrid models: every 128 tokens, deepest 4 kept


class BadRequest(ValueError):
    def __init__(self, detail):
        super().__init__(detail[0]["msg"])
        self.detail = detail


def _bad(loc, msg, typ="value_error"):
    return BadRequest([{"loc": ["body", *loc], "msg": msg, "type": typ}])


def choice_confidence(p):
    """TypeSafe's formula: how far the top probability sits above uniform."""
    if len(p) == 1:
        return 1.0
    u = 1.0 / len(p)
    return max(0.0, (max(p) - u) / (1.0 - u))


def score_confidence(p):
    """TypeSafe's formula: 1 - expected distance from the mode / that of a uniform distribution."""
    if len(p) == 1:
        return 1.0
    mode = max(range(len(p)), key=p.__getitem__)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (len(p) - 1) / 2
    umad = sum(abs(i - c) for i in range(len(p))) / len(p)
    return max(0.0, 1.0 - dist / umad)


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def validate(body):
    """Mirror of the official OpenAPI schema. Returns (state, questions)."""
    if not isinstance(body, dict):
        raise _bad([], "Input should be a valid dictionary", "dict_type")
    if "state" not in body:
        raise _bad(["state"], "Field required", "missing")
    state, qs = body["state"], body.get("questions")
    if not isinstance(state, (str, dict, list)):
        raise _bad(["state"], "Input should be a valid string, dictionary or list")
    if not isinstance(qs, dict) or not qs:
        raise _bad(["questions"], "Field required" if qs is None else "Dictionary should have at least 1 item",
                   "missing" if qs is None else "too_short")
    for qid, q in qs.items():
        t = q.get("type") if isinstance(q, dict) else None
        if t not in ("noul", "choice", "score"):
            raise _bad(["questions", qid], "Input tag should be 'noul', 'choice' or 'score'", "union_tag_invalid")
        crit = q.get("criteria")
        if t == "choice" and (not isinstance(crit, dict) or not 1 <= len(crit) <= MAX_OPTIONS):
            raise _bad(["questions", qid, t, "criteria"], f"choice criteria must map 1-{MAX_OPTIONS} options to descriptions")
        if t == "score" and (not isinstance(crit, list) or not MIN_LEVELS <= len(crit) <= MAX_LEVELS):
            raise _bad(["questions", qid, t, "criteria"], f"score criteria must be an ordered list of {MIN_LEVELS}-{MAX_LEVELS} levels")
        if t == "noul" and crit is not None and not isinstance(crit, dict):
            raise _bad(["questions", qid, t, "criteria"], "noul criteria must be an object with optional true/false")
    return state, qs


MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF", b"BM")


def _image_bytes(v):
    """Bytes of a data-URL / base64 image, None for ordinary text. A broken data URL is a 422."""
    url = v.startswith("data:image")
    if not url and len(v) < 256:
        return None
    try:
        data = base64.b64decode(v.split(",", 1)[1] if url else v, validate=True)
    except ValueError:
        data = b""
    if data[:4] in MAGIC or data[:3] in MAGIC:
        return data
    if url:
        raise _bad(["state"], "image data URL could not be decoded")
    return None


def split_image(state):
    """OpenJev convention: state.screenshot / state.image (data URL or base64 PNG/JPEG/...) rides along as an image."""
    if isinstance(state, dict):
        for k in IMAGE_KEYS:
            v = state.get(k)
            data = _image_bytes(v) if isinstance(v, str) else None
            if data:
                return {x: y for x, y in state.items() if x != k}, data
    return state, None


SPILL_MARGIN_MB, GTT_GROWTH_MB = 192, 512


def gpu_mem():
    """(vram_used, gtt_used, vram_total) in MB for the GPU with the most VRAM (the RX 7700S), or None."""
    best = None
    for d in glob.glob("/sys/class/drm/card*/device"):
        try:
            total = int(open(f"{d}/mem_info_vram_total").read()) >> 20
            if not best or total > best[2]:
                best = (int(open(f"{d}/mem_info_vram_used").read()) >> 20, int(open(f"{d}/mem_info_gtt_used").read()) >> 20, total)
        except (OSError, ValueError):
            continue
    return best


def spilled(before, after, vram_total):
    """True when an allocation overflowed VRAM into system RAM (GTT): the amdgpu driver allows it silently, and
    on a 14 GB laptop it ends in the kernel OOM-killing the desktop."""
    if not before or not after or not vram_total:
        return False
    return after[0] > vram_total - SPILL_MARGIN_MB or after[1] - before[1] > GTT_GROWTH_MB


class Engine:
    def __init__(self, model_path, mmproj=None, n_ctx=65536, calib=None, n_seq=8, name=None):
        self.name = name or os.path.basename(model_path).removesuffix(".gguf")
        self.calib = {**DEFAULT_CALIB, **(calib or {})}
        L.llama_backend_init()
        self._quiet = L.llama_log_callback(lambda level, text, data: None)
        if not os.environ.get("JEV_VERBOSE"):
            L.llama_log_set(self._quiet, None)
        mp = L.llama_model_default_params()
        mp.n_gpu_layers = int(os.environ.get("JEV_GPU_LAYERS", "-1"))
        m0 = gpu_mem()
        self.model = L.llama_model_load_from_file(model_path.encode(), mp)
        if not self.model:
            raise RuntimeError(f"cannot load {model_path}")
        m1 = gpu_mem()
        if spilled(m0, m1, m0 and m0[2]):
            L.llama_model_free(self.model)
            raise RuntimeError(f"{os.path.basename(model_path)} does not fit in free VRAM (it would spill into system RAM); "
                               "close other GPU programs or pick a smaller model")
        self.mtmd = None
        if mmproj:  # before the context, so the context-fit loop below budgets the projector's VRAM too
            import llama_cpp.mtmd_cpp as M
            self.M = M
            if not os.environ.get("JEV_VERBOSE"):
                M.mtmd_helper_log_set(self._quiet, None)
            mcp = M.mtmd_context_params_default()
            mcp.use_gpu, mcp.n_threads = True, os.cpu_count() // 2
            self.mtmd = M.mtmd_init_from_file(mmproj.encode(), self.model, mcp)
            if not self.mtmd or spilled(m0, gpu_mem(), m0 and m0[2]):
                raise RuntimeError(f"cannot load vision projector {mmproj} within free VRAM")
            m1 = gpu_mem()
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        n_ctx = min(n_ctx, L.llama_model_n_ctx_train(self.model) or n_ctx)
        self.n_seq = n_seq
        while True:  # halve the context until it fits in VRAM
            cp = L.llama_context_default_params()
            cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max = n_ctx, 2048, 1024, n_seq + 1 + N_CKPT
            cp.n_threads = cp.n_threads_batch = os.cpu_count() // 2
            cp.flash_attn_type = L.LLAMA_FLASH_ATTN_TYPE_ENABLED
            cp.type_k = cp.type_v = 8  # GGML_TYPE_Q8_0
            cp.kv_unified = True  # all sequences share the state's KV cells
            cp.no_perf = True
            self.ctx = L.llama_init_from_model(self.model, cp)
            if self.ctx and spilled(m1, gpu_mem(), m1 and m1[2]):  # fits only by eating system RAM: shrink instead
                L.llama_free(self.ctx)
                self.ctx = None
            if self.ctx or n_ctx <= 4096:
                break
            n_ctx //= 2
        if not self.ctx:
            L.llama_model_free(self.model)
            raise RuntimeError("cannot create a llama context that fits in VRAM")
        self.n_ctx, self.n_batch = L.llama_n_ctx(self.ctx), 2048
        self.mem = L.llama_get_memory(self.ctx)
        self.batch = L.llama_batch_init(self.n_batch, 0, 1)
        self.recurrent = L.llama_model_is_recurrent(self.model) or L.llama_model_is_hybrid(self.model)
        self.head, self.tail = self._template()
        self.letter_ids = [self._tok(c) for c in LETTERS]
        if any(len(t) != 1 for t in self.letter_ids) or len({t[0] for t in self.letter_ids}) != len(LETTERS):
            raise RuntimeError("tokenizer does not map each option letter to its own single token")
        self.letter_ids = [t[0] for t in self.letter_ids]
        self.pad = self._tok("\n")[0]
        self.cached, self.cached_text, self.ckpts = [], True, []  # prompt in sequence 0; checkpoints (n_past, seq)
        self.ckpt_ids = list(range(n_seq + 1, n_seq + 1 + N_CKPT))
        self.trace = None

    # ---------- tokens ----------
    def _tok(self, text, special=False, bos=False):
        b = text.encode()
        buf = (L.llama_token * (len(b) + 8))()
        n = L.llama_tokenize(self.vocab, b, len(b), buf, len(buf), bos, special)
        if n < 0:
            buf = (L.llama_token * -n)()
            n = L.llama_tokenize(self.vocab, b, len(b), buf, len(buf), bos, special)
        return list(buf[:n])

    def _template(self):
        """(head, tail) around the user content, rendered by llama.cpp from the model's own template."""
        tmpl = L.llama_model_chat_template(self.model, None)
        sentinel = "\x01JEV\x01"
        msg = (L.llama_chat_message * 1)(L.llama_chat_message(role=b"user", content=sentinel.encode()))
        buf = ctypes.create_string_buffer(4096)
        n = L.llama_chat_apply_template(tmpl, msg, 1, True, buf, len(buf))
        if n < 0:  # no usable template: plain text
            return "", "\nAnswer:"
        head, tail = buf.raw[:n].decode().split(sentinel)
        if "<think>" in (tmpl or b"").decode(errors="ignore") and "<think>" not in tail:
            tail += "<think>\n\n</think>\n\n"  # thinking off, as Qwen3.x's own template does
        return head, tail

    # ---------- KV plumbing ----------
    def _decode(self, items):
        """items: list of (token, pos, seq, want_logits). One llama_decode. Returns batch index per logits item."""
        b = self.batch
        for i, (t, p, s, want) in enumerate(items):
            b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = t, p, 1, want
            b.seq_id[i][0] = s
        b.n_tokens = len(items)
        rc = L.llama_decode(self.ctx, b)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed ({rc})")

    def _prefill(self, parts):
        """parts: list of token lists or ('img', bytes). Leaves exactly `parts` in sequence 0, reusing what it can:
        an extension of the cached prompt decodes only the tail; attention-only models truncate to the common prefix;
        hybrid/recurrent models (Qwen3.5) restore the deepest checkpoint inside the common prefix."""
        key = [t if isinstance(t, int) else hashlib.sha1(t[1]).hexdigest() for p in parts for t in (p if isinstance(p, list) else [p])]
        text = all(isinstance(p, list) for p in parts)
        self.prefilled, common = 0, 0
        for a, b in zip(key, self.cached):
            if a != b:
                break
            common += 1
        old, self.cached = self.cached, []  # invalid until this prefill completes: a failure must not leave stale reuse
        if old and common == len(old):
            start = common  # plain extension (also covers an identical prompt)
        elif text and self.cached_text and common and not self.recurrent:  # attention-only: truncate
            L.llama_memory_seq_rm(self.mem, 0, common, -1)
            start = self.n_past = common
        else:
            ck = max((c for c in self.ckpts if c[0] <= common), default=None) if text and self.cached_text else None
            for c in [c for c in self.ckpts if c is not ck and (not ck or c[0] > ck[0])]:
                L.llama_memory_seq_rm(self.mem, c[1], -1, -1)
                self.ckpts.remove(c)
            L.llama_memory_seq_rm(self.mem, 0, -1, -1)
            if ck:
                L.llama_memory_seq_cp(self.mem, ck[1], 0, -1, -1)
                start = self.n_past = ck[0]
            else:
                L.llama_memory_clear(self.mem, True)
                self.ckpts, start, self.n_past = [], 0, 0
        done = 0
        for p in parts:
            n = len(p) if isinstance(p, list) else 1
            if done + n <= start:
                done += n
                continue
            if isinstance(p, list):
                toks = p[max(0, start - done):]
                while toks:
                    step = self.n_batch
                    if text and self.recurrent:  # stop at the next checkpoint boundary
                        step = min(step, CKPT_EVERY - self.n_past % CKPT_EVERY)
                    chunk, toks = toks[:step], toks[step:]
                    self._decode([(t, self.n_past + j, 0, False) for j, t in enumerate(chunk)])
                    self.n_past += len(chunk)
                    self.prefilled += len(chunk)
                    if text and self.recurrent and toks and self.n_past % CKPT_EVERY == 0:
                        self._checkpoint()
            else:
                self._eval_image(p[1])
            done += n
        self.cached, self.cached_text = key, text
        return self.n_past

    def _checkpoint(self):
        """Snapshot sequence 0 at n_past into a spare sequence (recurrent state is copied, KV cells are shared)."""
        free = [s for s in self.ckpt_ids if s not in {c[1] for c in self.ckpts}]
        if not free:  # evict the shallowest
            old = min(self.ckpts)
            self.ckpts.remove(old)
            free = [old[1]]
        L.llama_memory_seq_rm(self.mem, free[0], -1, -1)
        L.llama_memory_seq_cp(self.mem, 0, free[0], -1, -1)
        self.ckpts.append((self.n_past, free[0]))

    def _eval_image(self, data):
        M = self.M
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        bmp = M.mtmd_helper_bitmap_init_from_buf(self.mtmd, buf, len(data), False)
        if not bmp:
            raise _bad(["state"], "image could not be decoded")
        chunks = M.mtmd_input_chunks_init()
        marker = M.mtmd_default_marker()
        txt = M.mtmd_input_text(text=marker, text_len=len(marker), add_special=False, parse_special=True)
        arr = (M.mtmd_bitmap_p_ctypes * 1)(bmp)
        try:
            if M.mtmd_tokenize(self.mtmd, chunks, ctypes.byref(txt), arr, 1) != 0:
                raise _bad(["state"], "image could not be tokenized")
            new_past = L.llama_pos(0)
            if M.mtmd_helper_eval_chunks(self.mtmd, self.ctx, chunks, self.n_past, 0, self.n_batch, False, ctypes.byref(new_past)) != 0:
                raise RuntimeError("image evaluation failed")
            self.n_past = new_past.value
        finally:
            M.mtmd_input_chunks_free(chunks)
            M.mtmd_bitmap_free(bmp)

    def _readouts(self, jobs):
        """jobs: list of (suffix tokens, n_options). Returns raw letter logits per job."""
        out = [None] * len(jobs)
        room = min(self.n_batch, self.n_ctx - self.n_past)
        i = 0
        while i < len(jobs):
            group, used = [], 0
            while i < len(jobs) and len(group) < self.n_seq and (len(group) + 1) * max(used, len(jobs[i][0])) <= room:
                group.append(i)
                used = max(used, len(jobs[i][0]))  # widest suffix; the group costs len(group) * used after padding
                i += 1
            if not group:
                raise _bad(["state"], "state and question do not fit the context window")
            items, last = [], []
            width = max(len(jobs[j][0]) for j in group)
            for s, j in enumerate(group, 1):
                L.llama_memory_seq_cp(self.mem, 0, s, -1, -1)
                toks = jobs[j][0]
                if self.recurrent:  # equal lengths let llama.cpp run all sequences in one equal-split ubatch
                    toks = [self.pad] * (width - len(toks)) + toks
                items += [(t, self.n_past + k, s, k == len(toks) - 1) for k, t in enumerate(toks)]
                last.append(len(items) - 1)
            self._decode(items)
            for j, bi in zip(group, last):
                lg = L.llama_get_logits_ith(self.ctx, bi)
                out[j] = [lg[self.letter_ids[k]] for k in range(jobs[j][1])]
            for s in range(1, len(group) + 1):
                L.llama_memory_seq_rm(self.mem, s, -1, -1)
        return out

    # ---------- prompts ----------
    def _suffix(self, instructions, opts):
        return self._tok(question(instructions, opts)) + self._tok(self.tail, special=True)

    def _orders(self, n, perms):
        orders = [list(range(n))]
        if perms >= 2 and n > 1:
            orders.append(orders[0][::-1])
        return orders

    def _dist_jobs(self, instructions, opts, perms):
        """Plan the readouts for one question. Returns (jobs, combine(logit lists) -> probs)."""
        T = self.calib["T"]
        bias = self.calib["bias"].get(str(len(opts))) or [0.0] * len(opts)  # fitted letter-position prior
        if len(opts) <= len(LETTERS):
            orders = self._orders(len(opts), perms)
            jobs = [(self._suffix(instructions, [opts[i] for i in o]), len(opts)) for o in orders]

            def combine(res):
                if self.trace is not None:  # calibration hook: raw letter logits per lettering
                    self.trace.append((orders, [list(z) for z in res]))
                acc = [0.0] * len(opts)
                for o, z in zip(orders, res):
                    for pos, i in enumerate(o):
                        acc[i] += _softmax([(v - b) / T for v, b in zip(z, bias)])[pos] / len(orders)
                return acc
            return jobs, combine
        return None, None  # >52 options: handled by _many

    def _many(self, instructions, opts, perms):
        """>52 options: readout per chunk, then over the chunk winners; composed into one distribution."""
        k = -(-len(opts) // len(LETTERS))
        size = -(-len(opts) // k)
        chunks = [opts[i:i + size] for i in range(0, len(opts), size)]
        planned = [self._dist_jobs(instructions, c, perms) for c in chunks]
        res = self._readouts([j for jobs, _ in planned for j in jobs])
        parts, at = [], 0
        for (jobs, comb) in planned:
            p = comb(res[at:at + len(jobs)])
            at += len(jobs)
            parts.append((p, max(range(len(p)), key=p.__getitem__)))
        jobs, comb = self._dist_jobs(instructions, [c[w] for c, (_, w) in zip(chunks, parts)], perms)
        pf = comb(self._readouts(jobs))
        raw = [pf[c] * pi / p[w] for c, (p, w) in enumerate(parts) for pi in p]
        s = sum(raw)
        return [v / s for v in raw], sum(len(j[0]) for j in jobs) + sum(len(j[0]) for jb, _ in planned for j in jb)

    # ---------- API ----------
    def systemone(self, body):
        t0 = time.perf_counter()
        state, qs = validate(body)
        state, image = split_image(state)
        if image is not None and not self.mtmd:
            raise _bad(["state"], "this model has no vision projector; start the server with a vision model")
        state_text = _text(state) if not (isinstance(state, dict) and not state and image) else "(see image)"
        parts = [self._tok(self.head, special=True, bos=True)]
        if image is not None:
            parts += [self._tok("Image:\n"), ("img", image)]
        parts.append(self._tok(("\n" if image is not None else "") + prefix(state_text)))
        perms = self.calib["perms"]

        # plan every question's readouts, then run them all against one prefill
        plans, n_state = {}, sum(len(p) for p in parts if isinstance(p, list))
        for qid, q in qs.items():
            t = q["type"]
            instr, opts = options(q)
            if len(opts) == 1:
                plans[qid] = (t, opts, None, None)
                continue
            jobs, comb = self._dist_jobs(instr, opts, perms)
            if jobs and n_state + max(len(j[0]) for j in jobs) > MAX_TOKENS:
                raise _bad(["state"], f"state plus question exceeds {MAX_TOKENS} tokens")
            plans[qid] = (t, opts, jobs, comb) if jobs else (t, opts, "many", instr)
        if n_state > MAX_TOKENS:
            raise _bad(["state"], f"state exceeds {MAX_TOKENS} tokens")
        widest = max((len(j[0]) for (_, _, jobs, _) in plans.values() if isinstance(jobs, list) for j in jobs), default=0)
        if n_state + widest + 64 > self.n_ctx:  # 64: headroom for the >52-option chunk prompts
            raise _bad(["state"], f"state plus question exceeds this server's {self.n_ctx}-token context (restart with a larger --ctx)")

        flat = [j for (_, _, jobs, _) in plans.values() if isinstance(jobs, list) for j in jobs]
        try:
            self._prefill(parts)
            res = self._readouts(flat)
        except Exception:
            for s in range(1, self.n_seq + 1):  # drop half-decoded question sequences
                L.llama_memory_seq_rm(self.mem, s, -1, -1)
            raise
        used = n_state + sum(len(j[0]) for j in flat)

        answers, at = {}, 0
        for qid, (t, opts, jobs, comb) in plans.items():
            if jobs is None:
                p = [1.0]
            elif jobs == "many":
                p, n = self._many(comb, opts, perms)
                used += n
            else:
                p = comb(res[at:at + len(jobs)])
                at += len(jobs)
            answers[qid] = self._answer(t, opts, p, qs[qid])
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": used, "output_tokens": len(answers)},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def _answer(self, t, opts, p, q):
        if t == "choice":
            best = max(range(len(p)), key=p.__getitem__)
            return {"type": "choice", "choice": opts[best][0],
                    "probabilities": {k: round(v, 4) for (k, _), v in zip(opts, p)},
                    "confidence": round(choice_confidence(p), 4)}
        if t == "score":
            return {"type": "score", "score": round(sum(i * v for i, v in enumerate(p)), 4),
                    "legend": {str(i): lv for i, lv in enumerate(q["criteria"])},
                    "probabilities": {str(i): round(v, 4) for i, v in enumerate(p)},
                    "confidence": round(score_confidence(p), 4)}
        py = min(max(p[0], 1e-6), 1 - 1e-6)
        z = math.log(py / (1 - py)) / self.calib["noul_T"] + self.calib["noul_b"]
        return {"type": "noul", "noul": round(1 / (1 + math.exp(-z)), 4)}

