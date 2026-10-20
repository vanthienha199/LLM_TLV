"""MM_RAG=1: retrieved reference notes appended to the worker prompt."""

import json
import os
import sys
import time

from . import config
from .workspace import snap

if config.ROUTER_DIR not in sys.path:
    sys.path.insert(0, config.ROUTER_DIR)

from rag import retrieve

Q1_HEAD = "\n# Reference notes (retrieved)\n\n"
Q2_HEAD = "\n\n# Reference notes for this failure (retrieved)\n\n"
_q1_cache = {}


def module_base():
    try:
        top = json.load(open(os.path.join(config.MDIR, "config.json"))).get("top", "")
    except Exception:
        top = ""
    return retrieve.canonical(os.path.basename(os.path.normpath(config.MDIR)), top)


def current_task():
    try:
        return json.load(open(os.path.join(config.MDIR, "status.json"))).get("task", "")
    except Exception:
        return ""


def log(rec):
    with open(os.path.join(config.MDIR, "rag_log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def augment(task, feedback, u, fb):
    idx = retrieve.index()
    base = module_base()
    wip = snap("wip.tlv")
    key = (task, wip, base)
    if key not in _q1_cache:
        _q1_cache[key] = idx.query1(task, wip, base)
    q1 = _q1_cache[key]
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "task": current_task(), "base": base,
           "q1": [[c["id"], s] for c, s in q1], "q2": []}
    if q1:
        u += Q1_HEAD + retrieve.render(q1)
    if feedback:
        q2 = idx.query2(feedback[-3000:], base, exclude={c["id"] for c, _ in q1})
        rec["q2"] = [[c["id"], s] for c, s in q2]
        if q2:
            fb += Q2_HEAD + retrieve.render(q2)
    rec["q1_chars"] = len(retrieve.render(q1)) if q1 else 0
    rec["q2_chars"] = len(fb) - len(fb.split(Q2_HEAD)[0]) if feedback else 0
    log(rec)
    return u, fb
