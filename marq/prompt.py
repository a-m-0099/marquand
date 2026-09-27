# the engine and train/distill.py both build prompts from here, so the trained models see the exact same text
import json

LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]


def text(v):
    return "" if v is None else v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def prefix(state_text):
    return f"State:\n{state_text}\n\n"


def options(q):
    t, instr, crit = q["type"], text(q.get("instructions")), q.get("criteria")
    if t == "choice":
        return instr, [(str(k), text(v)) for k, v in crit.items()]
    if t == "score":
        return instr + " Rate along the ordered levels below (lowest first).", [(str(i), text(v)) for i, v in enumerate(crit)]
    crit = crit or {}
    return instr, [("true", text(crit.get("true")) or "Yes, the statement is true."),
                   ("false", text(crit.get("false")) or "No, the statement is false.")]


def question(instructions, opts):
    lines = "\n".join(f"[{LETTERS[i]}] {k}" + (f": {d}" if d else "") for i, (k, d) in enumerate(opts))
    return f"Question: {instructions}\nOptions:\n{lines}\n\nAnswer with the letter of the best option only."
