#!/usr/bin/env python3
"""Self-contained tests for the judge context and the retry loop, driven
with fake providers so nothing leaves the machine. Each case sets up a
throwaway module dir, monkeypatches the provider call, and inspects the
prompt or the attempts.jsonl record the router wrote.

Run: python3 loop_test.py   (no pytest; prints "N/N cases passed", exit 1 on fail)
"""
import contextlib
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ["MM_DEEPSEEK_KEY_FILE"] = "/nonexistent/deepseek_key"
os.environ["MM_ANTHROPIC_KEY_FILE"] = "/nonexistent/anthropic_key"
os.environ["MM_PROVIDERS"] = "deepseek:2"
os.environ["MM_JUDGE"] = "0"
os.environ.pop("MM_ORDER", None)
os.environ.pop("MM_COMMON_GUIDE", None)

from lib import accounting, config, judge, runner, workspace
from lib import edits as edits_mod

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


EQY_BODY = "[gold]\nread_verilog -sv gold.sv\nprep -top serv_immdec_p2\n"


@case("judge: prompt includes every changed file, not only wip.tlv")
def _():
    with tempfile.TemporaryDirectory() as mdir:
        config.init(mdir)
        with open(os.path.join(mdir, "wip.tlv"), "w") as f:
            f.write("m4_TLV_version 1d\n")
        with open(os.path.join(mdir, "fev_full_p2.eqy"), "w") as f:
            f.write(EQY_BODY)
        seen = {}

        def fake_call(provider, user, tries=8, system=None):
            seen["prompt"] = user[0]
            return "VERDICT: PASS", {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}

        judge.call_with_retry = fake_call
        judge.track = lambda provider, u, **kw: 0.0
        passed, reason, cost = judge.judge("T", "task text", "before", "after",
                                           changed=["wip.tlv", "fev_full_p2.eqy"])
        p = seen["prompt"]
        assert passed is True
        assert "===FILE: fev_full_p2.eqy AFTER the task===\n" + EQY_BODY in p, p
        assert p.count("===FILE: wip.tlv") == 2, p
        assert "# Other files the worker changed in this attempt" in p, p


@case("judge: no extra block when only wip.tlv changed")
def _():
    with tempfile.TemporaryDirectory() as mdir:
        config.init(mdir)
        seen = {}

        def fake_call(provider, user, tries=8, system=None):
            seen["prompt"] = user[0]
            return "VERDICT: FAIL\nREASON: nope", {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}

        judge.call_with_retry = fake_call
        judge.track = lambda provider, u, **kw: 0.0
        passed, reason, cost = judge.judge("T", "task text", "before", "after", changed=["wip.tlv"])
        assert passed is False and reason == "nope"
        assert "Other files the worker changed" not in seen["prompt"]


TRUNC_REPLY = "===FILE: wip.tlv===\nm4_TLV_version 1d\n\\SV\n   module x(\n"
TRUNC_USAGE = {"in": 100, "out": 8000, "cache_read": 0, "cache_write": 0, "stop": "max_tokens"}
FULL_USAGE = {"in": 100, "out": 10, "cache_read": 0, "cache_write": 0, "stop": "end_turn"}


@case("loop: truncated reply skips apply, feeds targeted feedback, logs stop reason")
def _():
    with tempfile.TemporaryDirectory() as mdir:
        config.init(mdir)
        with open(os.path.join(mdir, "wip.tlv"), "w") as f:
            f.write("m4_TLV_version 1d\n")
        with open(os.path.join(mdir, "task_a.txt"), "w") as f:
            f.write("Do the thing.\n")
        with open(os.path.join(mdir, "order.json"), "w") as f:
            f.write('[["Task A", "task_a.txt"]]\n')
        os.environ["MM_ORDER"] = os.path.join(mdir, "order.json")
        replies = [(TRUNC_REPLY, dict(TRUNC_USAGE)), ("NO_CHANGE", dict(FULL_USAGE))]
        prompts = []

        def fake_call(provider, user, tries=8, system=None):
            prompts.append(user)
            return replies.pop(0)

        def no_apply(text):
            raise AssertionError("apply_files was called on a truncated reply")

        def no_fev():
            raise AssertionError("run_fev was called on a truncated reply")

        runner.call_with_retry = fake_call
        runner.apply_files = no_apply
        runner.run_fev = no_fev
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        assert not replies, "second attempt never ran"
        fb = prompts[1][2]
        assert "cut off by the output token limit after 8000 tokens" in fb, fb
        assert "stop reason: max_tokens" in fb, fb
        assert '"..." omission lines' in fb, fb
        assert "send it in full, with no commentary" in fb, fb
        recs = [json.loads(l) for l in open(os.path.join(mdir, "attempts.jsonl"))]
        assert [r["stop_reason"] for r in recs] == ["max_tokens", "end_turn"], recs
        assert "cut off by the output token limit" in recs[1]["feedback_in"], recs[1]
        assert accounting.stats[-1] == ("Task A", "deepseek (no-change)"), accounting.stats


def _agent_fallback_module(mdir):
    config.init(mdir)
    with open(os.path.join(mdir, "wip.tlv"), "w") as f:
        f.write("m4_TLV_version 1d\n")
    with open(os.path.join(mdir, "status.json"), "w") as f:
        f.write("{}\n")
    with open(os.path.join(mdir, "task_a.txt"), "w") as f:
        f.write("Do the thing.\n")
    with open(os.path.join(mdir, "order.json"), "w") as f:
        f.write('[["Task A", "task_a.txt"]]\n')
    os.environ["MM_ORDER"] = os.path.join(mdir, "order.json")


@case("fallback: deepseek:3,agent:1 calls the agent only after 3 raw failures, with the last FEV feedback")
def _():
    with tempfile.TemporaryDirectory() as mdir:
        _agent_fallback_module(mdir)
        good = ("===FILE: wip.tlv===\nm4_TLV_version 1d\n// raw edit\n===END===\n", dict(FULL_USAGE))
        replies = [good, good, (TRUNC_REPLY, dict(TRUNC_USAGE))]
        order = []
        seen = {}

        def fake_call(provider, user, tries=8, system=None):
            order.append(provider)
            return replies.pop(0)

        def fake_agent(task, feedback):
            order.append("agent")
            seen["feedback"] = feedback
            with open(os.path.join(mdir, "wip.tlv"), "a") as f:
                f.write("// agent edit\n")
            with open(os.path.join(mdir, "status.json"), "w") as f:
                f.write('{"tampered": true}\n')
            return "done", 1.0

        fev_results = [(False, "FEV FAILED: first mismatch"), (False, "FEV FAILED: second mismatch"),
                       (True, "ok")]
        saved = (runner.PROVIDERS, runner.call_with_retry, runner.run_agent_worker, runner.run_fev,
                 runner.revert, runner.enrich_feedback, runner.apply_files)
        runner.PROVIDERS = [("deepseek", 3), ("agent", 1)]
        runner.call_with_retry = fake_call
        runner.run_agent_worker = fake_agent
        runner.run_fev = lambda: fev_results.pop(0)
        runner.revert = lambda: None
        runner.enrich_feedback = lambda out: out
        runner.apply_files = edits_mod.apply_files
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main()
        finally:
            (runner.PROVIDERS, runner.call_with_retry, runner.run_agent_worker, runner.run_fev,
             runner.revert, runner.enrich_feedback, runner.apply_files) = saved
        assert order == ["deepseek", "deepseek", "deepseek", "agent"], order
        assert seen["feedback"] == "FEV FAILED: second mismatch", seen
        assert "tampered" not in open(os.path.join(mdir, "status.json")).read(), "harness file not restored"
        assert accounting.stats[-1] == ("Task A", "agent"), accounting.stats


@case("fallback: agent_feedback passes real feedback through, swaps truncation notes")
def _():
    from lib import providers
    trunc = providers.truncation_feedback(dict(TRUNC_USAGE))
    assert providers.agent_feedback("FEV FAILED", "older") == "FEV FAILED"
    assert providers.agent_feedback(trunc, "FEV FAILED") == "FEV FAILED"
    assert providers.agent_feedback(trunc, None) is None
    assert providers.agent_feedback(None, None) is None


@case("fallback: agent argv, prompt, key env and JSON cost parsing")
def _():
    from lib import providers
    with tempfile.TemporaryDirectory() as d:
        key = os.path.join(d, "key")
        with open(key, "w") as f:
            f.write("sk-test\n")
        saved = (config.AGENT_REFS, config.AGENT_KEY_FILE, config.AGENT_MAX_BUDGET, config.AGENT_EXTRA_ARGS)
        config.AGENT_EXTRA_ARGS = ["--bare"]
        config.AGENT_REFS = [d, os.path.join(d, "missing")]
        config.AGENT_KEY_FILE = key
        config.AGENT_MAX_BUDGET = "0.4"
        try:
            argv = providers.agent_argv()
            assert argv[argv.index("--add-dir") + 1] == d and argv.count("--add-dir") == 1, argv
            assert argv[argv.index("--max-budget-usd") + 1] == "0.4", argv
            assert argv[argv.index("--output-format") + 1] == "json", argv
            assert argv[2] == "--bare", argv
            assert providers.agent_env()["ANTHROPIC_API_KEY"] == "sk-test"
            p = providers._agent_prompt("Do X.", "FEV FAILED")
            assert "- " + d + "\n" in p and "missing" not in p, p
            assert p.index("# Reference material") < p.index("# Task") < p.index("FEV FAILED"), p
            config.AGENT_KEY_FILE = ""
            os.environ["ANTHROPIC_API_KEY"] = "leak"
            assert "ANTHROPIC_API_KEY" not in providers.agent_env()
            os.environ.pop("ANTHROPIC_API_KEY")
        finally:
            config.AGENT_REFS, config.AGENT_KEY_FILE, config.AGENT_MAX_BUDGET, config.AGENT_EXTRA_ARGS = saved
    out = json.dumps({"result": "Edited wip.tlv.", "total_cost_usd": 0.21, "num_turns": 7,
                      "subtype": "success", "usage": {"input_tokens": 5, "output_tokens": 900}})
    report, c, info = providers.parse_agent_output(out)
    assert report == "Edited wip.tlv." and c == 0.21 and info["turns"] == 7 and info["out"] == 900
    assert providers.parse_agent_output("plain text") == ("plain text", 0.0, {})


@case("fallback: NO_CHANGE before an agent is not cross-checked through the raw API")
def _():
    with tempfile.TemporaryDirectory() as mdir:
        _agent_fallback_module(mdir)
        calls = []

        def fake_call(provider, user, tries=8, system=None):
            calls.append(provider)
            if provider != "deepseek":
                raise AssertionError("raw API call for " + provider)
            return "NO_CHANGE", dict(FULL_USAGE)

        saved = (runner.PROVIDERS, runner.call_with_retry)
        runner.PROVIDERS = [("deepseek", 3), ("agent", 1)]
        runner.call_with_retry = fake_call
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main()
        finally:
            runner.PROVIDERS, runner.call_with_retry = saved
        assert calls == ["deepseek"], calls
        assert accounting.stats[-1] == ("Task A", "deepseek (no-change)"), accounting.stats


def main():
    fails = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:
            fails += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(CASES) - fails}/{len(CASES)} cases passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
