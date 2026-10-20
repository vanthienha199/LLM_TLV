#!/usr/bin/env python3
"""Build the retrieval store (store.json) used when MM_RAG=1.

Two kinds of chunks:
  kb    knowledge chunks from a TL-Verilog reference directory
  pair  failure-to-fix pairs mined from past router module directories

Usage:
  MM_RAG_SOURCES=<tlv_reference dir> python3 build_index.py \\
      --runs <dir holding module dirs> [--runs <dir> ...] [--out store.json]

The reference directory is expected to contain conversion_rulebook.txt,
00_MASTER_REFERENCE.md and notes/0[1-6]_*.md. Each --runs directory holds
module directories with history/NNN/{status.json,wip.tlv,config.json,judge.json}.
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import canonical

MIN_WORDS = 150
MAX_WORDS = 400
MAX_DIFF_LINES = 60
MAX_PAIRS_PER_TASK = 3
TINY_WORDS = 25
CATALOG_KEEP = ("## 3.", "## 4.")
CATALOG_SKIP_SUB = ("### 3.22",)
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
RULE_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def words(s):
    return len(s.split())


def split_long(text, limit=MAX_WORDS):
    paras = re.split(r"\n\s*\n", text.strip())
    out, cur = [], []
    for p in paras:
        if cur and words("\n\n".join(cur + [p])) > limit:
            out.append("\n\n".join(cur))
            cur = []
        if words(p) > limit:
            lines, buf = p.split("\n"), []
            for ln in lines:
                if buf and words("\n".join(buf + [ln])) > limit:
                    out.append("\n".join(buf))
                    buf = []
                buf.append(ln)
            if buf:
                cur.append("\n".join(buf))
            continue
        cur.append(p)
    if cur:
        out.append("\n\n".join(cur))
    return [o for o in out if o.strip()]


def md_sections(text):
    secs, path, cur_head, cur = [], [], "", []
    in_fence = False
    for ln in text.split("\n"):
        if ln.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(ln)
        if m:
            if cur and "\n".join(cur).strip():
                secs.append((cur_head, "\n".join(cur)))
            lvl = len(m.group(1))
            path = path[:lvl - 1] + [m.group(2).strip()]
            cur_head = " > ".join(path)
            cur = [ln]
        else:
            cur.append(ln)
    if cur and "\n".join(cur).strip():
        secs.append((cur_head, "\n".join(cur)))
    return secs


def catalog_filter(text):
    keep, on, skip_sub = [], False, False
    for ln in text.split("\n"):
        if ln.startswith("## "):
            on = ln.startswith(CATALOG_KEEP)
            skip_sub = False
        if ln.startswith("### "):
            skip_sub = ln.startswith(CATALOG_SKIP_SUB)
        if on and not skip_sub:
            keep.append(ln)
    return "\n".join(keep)


def md_chunks(rel, text):
    chunks, buf, head = [], [], None
    for h, body in md_sections(text):
        if head is None:
            head = h
        buf.append(body)
        if words("\n".join(buf)) >= MIN_WORDS:
            for piece in split_long("\n".join(buf)):
                chunks.append({"source": rel, "heading": head, "text": piece})
            buf, head = [], None
    if buf and "\n".join(buf).strip():
        tail = "\n".join(buf)
        if chunks and chunks[-1]["source"] == rel and words(chunks[-1]["text"] + tail) <= MAX_WORDS:
            chunks[-1]["text"] += "\n" + tail
        else:
            for piece in split_long(tail):
                chunks.append({"source": rel, "heading": head, "text": piece})
    return chunks


def rulebook_chunks(rel, text):
    chunks, section = [], ""
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        m = RULE_RE.match(s)
        if m:
            chunks.append({"source": rel, "heading": f"{section} rule {m.group(1)}".strip(),
                           "text": f"Rule {m.group(1)} ({section.title()}): {m.group(2)}"})
        elif s.isupper() or (s == s.upper() and any(c.isalpha() for c in s)):
            section = s
    return chunks


def knowledge(src):
    out = []
    rb = os.path.join(src, "conversion_rulebook.txt")
    if os.path.exists(rb):
        out += [dict(c, origin="rulebook") for c in rulebook_chunks("conversion_rulebook.txt", open(rb).read())]
    files = ["00_MASTER_REFERENCE.md"] + sorted(
        os.path.join("notes", f) for f in os.listdir(os.path.join(src, "notes"))
        if re.match(r"0[1-6]_.*\.md$", f))
    for rel in files:
        p = os.path.join(src, rel)
        if not os.path.exists(p):
            continue
        text = open(p).read()
        if rel.endswith("06_examples_catalog.md"):
            text = catalog_filter(text)
        out += [dict(c, origin=os.path.basename(rel)) for c in md_chunks(rel, text)
                if words(c["text"]) >= TINY_WORDS]
    for i, c in enumerate(out):
        c["id"] = f"kb:{c['source']}#{i}"
        c["kind"] = "kb"
    return out


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def checkpoints(mdir):
    hist = os.path.join(mdir, "history")
    if not os.path.isdir(hist):
        return []
    out = []
    for d in sorted((d for d in os.listdir(hist) if d.isdigit()), key=int):
        cp = os.path.join(hist, d)
        st = load_json(os.path.join(cp, "status.json"))
        jd = load_json(os.path.join(cp, "judge.json"))
        fev = str(st.get("fev.sh", ""))
        ok = fev.startswith("0:")
        jfail = jd.get("verdict") == "FAIL"
        out.append({"n": d, "task": st.get("task", ""), "fev": fev, "fev_ok": ok,
                    "judge_fail": jfail, "judge_reason": jd.get("reason", ""),
                    "wip": read(os.path.join(cp, "wip.tlv")),
                    "top": load_json(os.path.join(cp, "config.json")).get("top", "")})
    return out


def compact_diff(a, b):
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "failing/wip.tlv",
                                  "passing/wip.tlv", n=2, lineterm=""))
    if len(d) > MAX_DIFF_LINES:
        d = d[:MAX_DIFF_LINES] + [f"... (diff truncated, {len(d) - MAX_DIFF_LINES} more lines)"]
    return "\n".join(d)


def pairs_for_module(mdir):
    name = os.path.basename(os.path.normpath(mdir))
    cps = checkpoints(mdir)
    out = []
    for i, cp in enumerate(cps):
        failed = (not cp["fev_ok"] and cp["fev"] not in ("", "none")) or cp["judge_fail"]
        if not failed or not cp["task"] or not cp["wip"]:
            continue
        nxt = None
        for later in cps[i + 1:]:
            if later["task"] != cp["task"]:
                break
            if later["fev_ok"] and not later["judge_fail"]:
                nxt = later
                break
        if nxt is None or not nxt["wip"]:
            continue
        diff = compact_diff(cp["wip"], nxt["wip"])
        if not diff:
            continue
        if cp["judge_fail"] and cp["fev_ok"]:
            failure = "Reviewer judged the task goal not achieved: " + cp["judge_reason"][:600]
        else:
            failure = cp["fev"]
        out.append({"module": name, "base": canonical(name, cp["top"] or nxt["top"]), "task": cp["task"],
                    "fail_ckpt": cp["n"], "fix_ckpt": nxt["n"], "failure": failure, "diff": diff})
    return out


def mine(run_dirs):
    seen, per_task, out = set(), {}, []
    for rd in run_dirs:
        for m in sorted(os.listdir(rd)):
            mdir = os.path.join(rd, m)
            if not os.path.isdir(os.path.join(mdir, "history")):
                continue
            for p in reversed(pairs_for_module(mdir)):
                h = hashlib.sha1((p["task"] + "\n" + p["diff"]).encode()).hexdigest()[:12]
                if h in seen:
                    continue
                key = (p["module"], p["task"])
                if per_task.get(key, 0) >= MAX_PAIRS_PER_TASK:
                    continue
                seen.add(h)
                per_task[key] = per_task.get(key, 0) + 1
                p["id"] = f"pair:{p['module']}:{p['fail_ckpt']}->{p['fix_ckpt']}"
                p["kind"] = "pair"
                p["source"] = os.path.join(os.path.basename(os.path.normpath(rd)), p["module"])
                p["heading"] = f"Past fix: {p['task']} ({p['module']})"
                p["text"] = (f"Task: {p['task']}\nFailure: {p['failure']}\n"
                             f"Change from the failing wip.tlv to the next passing one:\n{p['diff']}")
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=os.environ.get("MM_RAG_SOURCES", ""))
    ap.add_argument("--runs", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "store.json"))
    a = ap.parse_args()
    if not a.sources or not os.path.isdir(a.sources):
        sys.exit("set MM_RAG_SOURCES or --sources to the tlv_reference directory")
    kb = knowledge(a.sources)
    pairs = mine(a.runs)
    counts = {}
    for c in kb:
        counts[c["origin"]] = counts.get(c["origin"], 0) + 1
    counts["pairs"] = len(pairs)
    keep = ("id", "kind", "source", "heading", "text", "module", "base", "task")
    store = {"counts": counts, "chunks": [{k: c[k] for k in keep if k in c} for c in kb + pairs]}
    with open(a.out, "w") as f:
        json.dump(store, f, indent=0, sort_keys=True)
    print(json.dumps(counts, indent=1))
    print(f"wrote {len(store['chunks'])} chunks to {a.out}")


if __name__ == "__main__":
    main()
