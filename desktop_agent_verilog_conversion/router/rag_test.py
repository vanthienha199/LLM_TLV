#!/usr/bin/env python3
"""Self-contained tests for the MM_RAG retrieval path (rag/ and lib/rag_hook.py).

Run: python3 rag_test.py   (no pytest; prints "N/N cases passed", exit 1 on fail)
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.pop("MM_RAG", None)
os.environ.pop("MM_COMMON_GUIDE", None)

from lib import config, prompts, workspace
from rag import retrieve

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


MAIN_BUILD_USER = '''
def build_user(task, feedback=None):
    u = "# Task\\n\\n" + task + "\\n"
    for n in context_files():
        u += f"\\n===FILE: {n}===\\n" + snap(n) + "\\n===END===\\n"
    fb = ""
    if feedback:
        fb = ("\\n# Previous attempt FAILED verification. Tool output:\\n\\n" + feedback[-3000:] +
              "\\n\\nFix the problem and reply with the complete corrected files. If the "
              "remaining constructs genuinely cannot be converted for a specific technical "
              "reason, it is acceptable to reply NO_CHANGE followed by a "
              "===JUSTIFICATION===/===END=== block naming the exact constructs and why; "
              "the reviewing agent will weigh it on its merits.")
    return COMMON, u, fb
'''


def main_build_user():
    ns = {"COMMON": prompts.COMMON, "context_files": workspace.context_files, "snap": workspace.snap}
    exec(MAIN_BUILD_USER, ns)
    return ns["build_user"]


WIP = ("\\m5_TLV_version 1d: tl-x.org\n\\m5\n   use(m5-1.0)\n\\SV\n   module serv_rf_if();\n"
       "\\TLV\n   |default\n      @0\n         $rd_wen = $rd_wen_in & (| $rd_waddr);\n"
       "         \\SV_plus\n            assign o_wen0 = $cnt_en & $rd_wen;\n\\SV\n   endmodule\n")


def make_module(mdir, top="serv_rf_if"):
    files = {"wip.tlv": WIP, "fev.eqy": "[gold]\nread_verilog -sv -formal feved.sv\n",
             "fev_full.eqy": "[gold]\nread_verilog -sv -formal prepared.sv\n",
             "fev_full_WITH_CSR_0.eqy": "[script]\nchparam -set WITH_CSR 0 serv_rf_if\n",
             "config.json": json.dumps({"top": top}), "tracker.md": "# Tracker\n",
             "status.json": json.dumps({"task": "Non-vector Signals"})}
    for n, body in files.items():
        with open(os.path.join(mdir, n), "w") as f:
            f.write(body)


def tiny_store(path):
    chunks = [
        {"id": "kb:a#0", "kind": "kb", "source": "a", "heading": "State",
         "text": "Use >>1$sig to read the previous cycle value. \\SV_plus keeps Verilog."},
        {"id": "kb:b#1", "kind": "kb", "source": "b", "heading": "M5",
         "text": "m5_var and m5_if_eq_block define configurations for |default pipelines."},
        {"id": "pair:serv_rf_if_fresh:003->004", "kind": "pair", "source": "x/serv_rf_if_fresh",
         "module": "serv_rf_if_fresh", "base": "serv_rf_if", "task": "Non-vector Signals",
         "heading": "Past fix", "text": "Failure: BAD-SCOPE $rd_wen in \\SV_plus"},
        {"id": "pair:serv_rf_if_e1:005->006", "kind": "pair", "source": "x/serv_rf_if_e1",
         "module": "serv_rf_if_e1", "base": "", "task": "Non-vector Signals",
         "heading": "Past fix", "text": "Failure: BAD-SCOPE $rd_wen in \\SV_plus"},
        {"id": "pair:serv_rf_ram_c2:024->027", "kind": "pair", "source": "x/serv_rf_ram_c2",
         "module": "serv_rf_ram_c2", "base": "serv_rf_ram", "task": "Non-vector Signals",
         "heading": "Past fix", "text": "Failure: BAD-SCOPE $rdata in \\SV_plus"},
        {"id": "pair:serv_alu:010->011", "kind": "pair", "source": "x/serv_alu",
         "module": "serv_alu", "base": "serv_alu", "task": "Non-vector Signals",
         "heading": "Past fix", "text": "Failure: BAD-SCOPE $rd_wen"},
        {"id": "pair:serv_ctrl:002->003", "kind": "pair", "source": "x/serv_ctrl",
         "module": "serv_ctrl", "base": "serv_ctrl", "task": "Reset and Clock",
         "heading": "Past fix", "text": "Failure: BAD-SCOPE"},
        {"id": "pair:serv_csr:004->005", "kind": "pair", "source": "x/serv_csr",
         "module": "serv_csr", "base": "serv_csr", "task": "Reset and Clock",
         "heading": "Past fix", "text": "Failure: BAD-SCOPE again"},
        {"id": "kb:c#2", "kind": "kb", "source": "c", "heading": "Viz",
         "text": "viz_js renders boxes."},
    ]
    with open(path, "w") as f:
        json.dump({"counts": {}, "chunks": chunks}, f)


@case("tokens keep TL-Verilog constructs intact")
def _():
    t = retrieve.tokens("\\SV_plus m5_var(cond, 1) >>1$sig |pipe /scope[3:0] BAD-SCOPE m5+mac(/_top) $$out ?$valid @1 <>0")
    for want in ("\\SV_plus", "m5_var", ">>1", "$sig", "|pipe", "/scope[3:0]", "BAD-SCOPE",
                 "m5+mac", "$$out", "?$valid", "@1", "<>0", "sig", "pipe", "scope"):
        assert want in t, (want, t)


@case("canonical base folds run variants onto the module")
def _():
    c = retrieve.canonical
    assert c("serv_rf_if_fresh") == "serv_rf_if"
    assert c("serv_rf_if_e1") == "serv_rf_if"
    assert c("serv_rf_if") == "serv_rf_if"
    assert c("serv_rf_ram_if_x") == "serv_rf_ram_if"
    assert c("serv_rf_ram_c2") == "serv_rf_ram"
    assert c("immdec_probe2") == "serv_immdec"
    assert c("serv_rf_if_ragB_1", "serv_rf_if") == "serv_rf_if"


@case("leakage: pairs from the module under test and its variants never come back")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "store.json")
        tiny_store(p)
        idx = retrieve.Index(p)
        ids = [c["id"] for c, _ in idx.query2("BAD-SCOPE $rd_wen \\SV_plus", "serv_rf_if", k=10)]
        assert "pair:serv_rf_ram_c2:024->027" in ids, ids
        assert not any("serv_rf_if" in i for i in ids), ids
        ids = [c["id"] for c, _ in idx.query2("BAD-SCOPE $rd_wen \\SV_plus", "serv_rf_ram", k=10)]
        assert not any("serv_rf_ram" in i for i in ids), ids


@case("MM_RAG unset: build_user is byte-identical to origin/main")
def _():
    ref = main_build_user()
    with tempfile.TemporaryDirectory() as mdir:
        make_module(mdir)
        config.init(mdir)
        for fb in (None, "", "short failure", "x" * 5000 + "ERROR tail"):
            assert prompts.build_user("Summary: task text\n", fb) == ref("Summary: task text\n", fb)
        assert not os.path.exists(os.path.join(mdir, "rag_log.jsonl"))


@case("MM_RAG=1: notes go after the files and after the feedback, prefixes unchanged, log written")
def _():
    from lib import rag_hook
    ref = main_build_user()
    with tempfile.TemporaryDirectory() as mdir:
        p = os.path.join(mdir, "store.json")
        tiny_store(p)
        make_module(mdir)
        config.init(mdir)
        old = retrieve._INDEX
        retrieve._INDEX = retrieve.Index(p)
        os.environ["MM_RAG"] = "1"
        try:
            c0, u0, f0 = ref("Summary: convert \\SV_plus signals\n", None)
            c1, u1, f1 = prompts.build_user("Summary: convert \\SV_plus signals\n", None)
            assert c1 == c0 and f1 == f0 == "", (len(c1), len(c0), f1)
            assert u1.startswith(u0) and rag_hook.Q1_HEAD in u1[len(u0):], u1[len(u0)-50:]
            c2, u2, f2 = prompts.build_user("Summary: convert \\SV_plus signals\n", "BAD-SCOPE $rd_wen")
            assert u2 == u1, "stable part must not change across retries"
            r2 = ref("Summary: convert \\SV_plus signals\n", "BAD-SCOPE $rd_wen")[2]
            assert f2.startswith(r2) and rag_hook.Q2_HEAD in f2[len(r2):], open(os.path.join(mdir, "rag_log.jsonl")).read()
            recs = [json.loads(l) for l in open(os.path.join(mdir, "rag_log.jsonl"))]
            assert len(recs) == 2 and recs[0]["task"] == "Non-vector Signals"
            assert recs[1]["base"] == "serv_rf_if"
            assert not any("serv_rf_if" in i for i, _ in recs[1]["q1"] + recs[1]["q2"])
            q1ids = {i for i, _ in recs[1]["q1"]}
            assert not q1ids & {i for i, _ in recs[1]["q2"]}
        finally:
            os.environ.pop("MM_RAG", None)
            retrieve._INDEX = old


@case("committed store loads and every pair carries module and base")
def _():
    idx = retrieve.Index()
    pairs = [c for c in idx.chunks if c["kind"] == "pair"]
    assert pairs and all(c.get("module") and c.get("base") for c in pairs)
    assert len({c["id"] for c in idx.chunks}) == len(idx.chunks)


def main():
    passed = 0
    for name, fn in CASES:
        try:
            fn()
            passed += 1
            print(f"  ok   {name}")
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"{passed}/{len(CASES)} cases passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
