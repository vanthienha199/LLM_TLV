"""Worker invocation: the deepseek and claude API calls (with prompt caching
and network retry) and the Claude Code agent worker subprocess."""

import json
import os
import subprocess
import time
import urllib.request

from . import accounting, config
from .prompts import AGENT_PREAMBLE, SYSTEM


def call_with_retry(provider, user, tries=8, system=None):
    for k in range(tries):
        try:
            return call(provider, user, system=system)
        except Exception as e:
            wait = min(60, 15 * (k + 1))
            print(f"    (network error {type(e).__name__}, retry {k+1}/{tries} in {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError("network retries exhausted")


def call(provider, user, system=None):
    # user is (common, stable, fb) from build_user, (stable, fb) from the
    # judge path, or a bare string. Common goes first so both providers can
    # cache the cross-task prefix.
    if isinstance(user, tuple):
        common, stable, fb = user if len(user) == 3 else ("", user[0], user[1])
    else:
        common, stable, fb = "", user, ""
    system = system or SYSTEM
    if provider == "deepseek":
        key = open(os.path.expanduser(os.environ.get("MM_DEEPSEEK_KEY_FILE", "~/.secrets/deepseek_key"))).read().strip()
        req = urllib.request.Request("https://api.deepseek.com/chat/completions",
            data=json.dumps({"model": config.MODEL_NAME["deepseek"],
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": common + stable + fb}],
                "max_tokens": config.DEEPSEEK_MAX_TOKENS, "temperature": 0.2}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key})
        d = json.load(urllib.request.urlopen(req, timeout=300))
        u = d.get("usage", {})
        usage = {"in": u.get("prompt_cache_miss_tokens", u.get("prompt_tokens", 0)),
                 "out": u.get("completion_tokens", 0),
                 "cache_read": u.get("prompt_cache_hit_tokens", 0),
                 "cache_write": 0,
                 "stop": d["choices"][0].get("finish_reason") or ""}
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content:
            # deepseek v4 can burn its whole budget on reasoning -> empty content
            print(f"    (deepseek empty content, finish_reason={d['choices'][0].get('finish_reason')})", flush=True)
            content = msg.get("reasoning_content") or ""
        return content, usage
    else:
        key = open(os.path.expanduser(os.environ.get("MM_ANTHROPIC_KEY_FILE", "~/.secrets/anthropic_key"))).read().strip()
        content = []
        if common:
            content.append({"type": "text", "text": common, "cache_control": {"type": "ephemeral"}})
        content.append({"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}})
        if fb:
            content.append({"type": "text", "text": fb})
        req = urllib.request.Request("https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": config.MODEL_NAME["claude"], "max_tokens": 8000, "system": system,
                "messages": [{"role": "user", "content": content}]}).encode(),
            headers={"Content-Type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"})
        d = json.load(urllib.request.urlopen(req, timeout=300))
        u = d.get("usage", {})
        usage = {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0),
                 "cache_read": u.get("cache_read_input_tokens", 0),
                 "cache_write": u.get("cache_creation_input_tokens", 0),
                 "stop": d.get("stop_reason") or ""}
        return d["content"][0]["text"], usage


# The provider's own stop signal: Anthropic reports stop_reason "max_tokens"
# and OpenAI-style APIs (deepseek) finish_reason "length" when the reply hit
# the output budget mid-file. A cut reply is recoverable, not malformed, so
# the loop must never hand it to apply_files.
def truncated(usage):
    return usage.get("stop") in ("max_tokens", "length")


# Feedback for the attempt after a truncated reply: say where it was cut and
# how to fit the same edit, so the model does not just resend the same
# oversized reply and lose another attempt.
TRUNCATION_PREFIX = "Your previous reply was cut off by the output token limit"


def truncation_feedback(usage):
    return (f"{TRUNCATION_PREFIX} after "
            f"{usage.get('out', 0)} tokens (stop reason: {usage.get('stop')}), so nothing "
            "was applied. Resend the same edit, but use \"...\" omission lines for every "
            "unchanged region of each file so the reply fits. If a whole file changes, "
            "send it in full, with no commentary or explanation around the file blocks.")


# Feedback the agent sees. A truncation note tells a raw API worker how to
# shorten its reply format, which means nothing to an agent editing files in
# place, so after a truncated raw attempt the agent gets the last real
# verification failure of this task instead (or none if there was none).
def agent_feedback(feedback, fev_feedback):
    if feedback and feedback.startswith(TRUNCATION_PREFIX):
        return fev_feedback
    return feedback


def _agent_prompt(task_text, feedback):
    prompt = AGENT_PREAMBLE
    refs = [d for d in config.AGENT_REFS if os.path.isdir(d)]
    if refs:
        prompt += ("# Reference material\n\nTL-Verilog and M5 reference notes, conversion rules, "
                   "and examples are in these directories (read-only for you). Search them with "
                   "Grep/Glob and read what you need before and while editing:\n"
                   + "".join(f"- {d}\n" for d in refs) + "\n")
    prompt += "# Task\n\n" + task_text + "\n"
    if feedback:
        prompt += ("\n# Previous attempt FAILED verification. Tool output:\n\n"
                   + feedback[-3000:] + "\n\nFix the problem by editing the files.\n")
    return prompt


def agent_argv():
    argv = [config.AGENT_CMD, "-p", *config.AGENT_EXTRA_ARGS, "--model", config.AGENT_MODEL,
            "--max-turns", str(config.AGENT_MAX_TURNS),
            "--output-format", "json",
            "--permission-mode", "acceptEdits",
            "--allowedTools", "Read,Edit,Write,Grep,Glob"]
    for d in config.AGENT_REFS:
        if os.path.isdir(d):
            argv += ["--add-dir", d]
    if config.AGENT_MAX_BUDGET:
        argv += ["--max-budget-usd", str(config.AGENT_MAX_BUDGET)]
    return argv


def agent_env():
    # Default: the logged-in subscription, so any API key in the environment
    # is removed. MM_AGENT_KEY_FILE switches the agent to that API key.
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    if config.AGENT_KEY_FILE:
        env["ANTHROPIC_API_KEY"] = open(os.path.expanduser(config.AGENT_KEY_FILE)).read().strip()
    return env


def parse_agent_output(stdout):
    # --output-format json prints one result object; fall back to raw text.
    try:
        d = json.loads(stdout)
    except (ValueError, TypeError):
        return (stdout or ""), 0.0, {}
    u = d.get("usage") or {}
    info = {"turns": d.get("num_turns", 0), "subtype": d.get("subtype", ""),
            "is_error": bool(d.get("is_error")),
            "in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0),
            "cache_read": u.get("cache_read_input_tokens", 0),
            "cache_write": u.get("cache_creation_input_tokens", 0)}
    return str(d.get("result") or d.get("subtype") or ""), float(d.get("total_cost_usd") or 0.0), info


def run_agent_worker(task_text, feedback):
    prompt = _agent_prompt(task_text, feedback)
    t_start = time.time()
    cost, info = 0.0, {}
    try:
        r = subprocess.run(agent_argv(), input=prompt, capture_output=True, text=True,
                           timeout=config.AGENT_TIMEOUT, cwd=config.MDIR, env=agent_env())
        report, cost, info = parse_agent_output(r.stdout)
        if r.returncode != 0 and r.stderr:
            report += "\n[stderr] " + r.stderr
    except subprocess.TimeoutExpired:
        report = f"[agent timed out after {config.AGENT_TIMEOUT}s]"
        info = {"subtype": "timeout"}
    wall = time.time() - t_start
    accounting.track_agent(cost, info, wall)
    return report.strip(), wall
