"""Cost and cache-usage tracking, per-provider spend caps, run summary."""

import json
import os
import sys
import time

from . import config
from .config import MAX_COST

cost = {"deepseek": 0.0, "claude": 0.0, "agent": 0.0}
cache_totals = {"read": 0, "write": 0, "in": 0}


def track(provider, u, role="worker"):
    # Cache pricing per the official price sheets: anthropic cache write is
    # 1.25x input and cache read 0.1x input; deepseek cache hit ~0.1x miss.
    if provider == "deepseek":
        c = (u["in"]*0.14 + u["cache_read"]*0.014 + u["out"]*0.28)/1e6
    else:
        c = (u["in"]*3.0 + u["cache_write"]*3.75 + u["cache_read"]*0.30 + u["out"]*15.0)/1e6
    cost[provider] += c
    log_usage(provider, role, u, c)
    cache_totals["read"] += u["cache_read"]
    cache_totals["write"] += u["cache_write"]
    cache_totals["in"] += u["in"]
    if cost[provider] > MAX_COST[provider]:
        print(f"!!! COST CAP: {provider} ${cost[provider]:.2f} > ${MAX_COST[provider]:.2f}, stopping safely")
        print_summary()
        sys.exit(2)
    return c


def log_usage(provider, role, u, c):
    if not config.MDIR:
        return
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "provider": provider, "role": role,
           "in": u.get("in", 0), "out": u.get("out", 0), "cache_read": u.get("cache_read", 0),
           "cache_write": u.get("cache_write", 0), "stop": u.get("stop", ""), "cost_usd": round(c, 6)}
    with open(os.path.join(config.MDIR, "usage.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def cache_str(u):
    return f"cache r{u['cache_read']}/w{u['cache_write']}/u{u['in']}"


stats = []
t0 = time.time()
agent_wall = [0.0]


def print_summary():
    print("\n===== RUN SUMMARY =====")
    for t, u in stats:
        print(f"  {u:36} {t}")
    print(f"cost: deepseek ${cost['deepseek']:.4f} + claude ${cost['claude']:.4f} + agent ${cost['agent']:.4f} = ${sum(cost.values()):.4f}")
    tot = cache_totals["read"] + cache_totals["in"]
    pct = 100.0 * cache_totals["read"] / tot if tot else 0.0
    print(f"cache: read {cache_totals['read']} / write {cache_totals['write']} / uncached {cache_totals['in']} (hit {pct:.0f}%)")
    print(f"wall clock: {(time.time()-t0)/60:.1f} min")
    if agent_wall[0]:
        print(f"agent worker wall: {agent_wall[0]/60:.1f} min")


# Agent worker accounting: cost is what the claude CLI reports (0 on the
# subscription), logged to usage.jsonl and held to MM_MAX_COST_AGENT.
def track_agent(c, info, wall):
    cost["agent"] += c
    agent_wall[0] += wall
    u = {k: info.get(k, 0) for k in ("in", "out", "cache_read", "cache_write")}
    u["stop"] = info.get("subtype", "")
    if config.MDIR:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "provider": "agent", "role": "worker",
               **u, "turns": info.get("turns", 0), "wall_s": round(wall, 1), "cost_usd": round(c, 6)}
        with open(os.path.join(config.MDIR, "usage.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    if cost["agent"] > MAX_COST["agent"]:
        print(f"!!! COST CAP: agent ${cost['agent']:.2f} > ${MAX_COST['agent']:.2f}, stopping safely")
        print_summary()
        sys.exit(2)
