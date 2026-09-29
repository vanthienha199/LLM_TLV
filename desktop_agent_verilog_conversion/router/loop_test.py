#!/usr/bin/env python3
"""Self-contained tests for the judge context and the retry loop, driven
with fake providers so nothing leaves the machine. Each case sets up a
throwaway module dir, monkeypatches the provider call, and inspects the
prompt or the attempts.jsonl record the router wrote.

Run: python3 loop_test.py   (no pytest; prints "N/N cases passed", exit 1 on fail)
"""
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

from lib import config, judge

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
        judge.track = lambda provider, u: 0.0
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
        judge.track = lambda provider, u: 0.0
        passed, reason, cost = judge.judge("T", "task text", "before", "after", changed=["wip.tlv"])
        assert passed is False and reason == "nope"
        assert "Other files the worker changed" not in seen["prompt"]


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
