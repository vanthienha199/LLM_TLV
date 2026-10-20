"""BM25 retrieval over the store built by build_index.py (stdlib only)."""

import json
import math
import os
import re

STORE = os.environ.get("MM_RAG_STORE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "store.json"))
K1 = 1.2
B = 0.75
PAIR_BOOST = 1.5
Q1_K = 4
Q2_K = 3
WIP_TERMS = 40

SERV_MODULES = ("serv_aligner", "serv_alu", "serv_bufreg", "serv_bufreg2", "serv_compdec",
                "serv_csr", "serv_ctrl", "serv_debug", "serv_decode", "serv_decoder",
                "serv_immdec", "serv_mem_if", "serv_rf", "serv_rf_if", "serv_rf_ram",
                "serv_rf_ram_if", "serv_rf_top", "serv_state", "serv_synth_wrapper", "serv_top")

TOKEN_RE = re.compile(
    r"\\[A-Za-z_][A-Za-z0-9_]*"
    r"|m[45]\+[A-Za-z_][A-Za-z0-9_]*"
    r"|m[45]_[A-Za-z0-9_]+"
    r"|<<\d+|>>\d+|<>\d+"
    r"|\?\$\$?[A-Za-z_][A-Za-z0-9_]*"
    r"|\$\$?[A-Za-z_][A-Za-z0-9_]*"
    r"|\|[A-Za-z_][A-Za-z0-9_]*"
    r"|/[A-Za-z_][A-Za-z0-9_]*(?:\[\*?\d*:?\d*\])?"
    r"|@\d+"
    r"|\*[A-Za-z_][A-Za-z0-9_]*"
    r"|[A-Z][A-Z0-9]+(?:[-_][A-Z0-9]+)+"
    r"|[A-Za-z_][A-Za-z0-9_]*"
)
STOP = set("""a an and are as at be been but by can do does each for from has have if in into is it
its may must no not of on one or so such that the their them then there these this those to use
used uses using was we were what when which while will with without you your all any also only
should would could than other more most same new file files line lines code task""".split())


def tokens(text):
    out = []
    for t in TOKEN_RE.findall(text):
        if t[0].isalpha() or t[0] == "_":
            if re.match(r"^[A-Z][A-Z0-9]+(?:[-_][A-Z0-9]+)+$", t):
                out.append(t)
                continue
            lw = t.lower()
            if len(lw) < 2 or lw in STOP:
                continue
            out.append(lw)
            continue
        out.append(t)
        bare = re.sub(r"^[^A-Za-z_]+", "", t).lower()
        bare = re.sub(r"\[.*$", "", bare)
        if t[0] in "$|/?*" and len(bare) >= 2 and bare not in STOP:
            out.append(bare)
    return out


def canonical(name, top=""):
    if top:
        return top
    n = name if name.startswith("serv_") else "serv_" + name
    best = ""
    for m in SERV_MODULES:
        if (n == m or n.startswith(m + "_")) and len(m) > len(best):
            best = m
    return best or name


class Index:
    def __init__(self, path=STORE):
        self.chunks = json.load(open(path))["chunks"]
        self.tf, self.dl = [], []
        df = {}
        for c in self.chunks:
            toks = tokens(c.get("heading", "") + "\n" + c["text"])
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            self.dl.append(len(toks) or 1)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        n = len(self.chunks)
        self.avgdl = sum(self.dl) / max(1, n)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def allowed(self, c, base):
        if c["kind"] != "pair":
            return True
        return canonical(c.get("module", ""), c.get("base", "")) != base and \
            canonical(c.get("module", "")) != base

    def score(self, i, qterms):
        tf, dl, s = self.tf[i], self.dl[i], 0.0
        for t in qterms:
            f = tf.get(t)
            if f:
                s += self.idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / self.avgdl))
        return s

    def search(self, qterms, base, k, exclude=(), pair_boost=1.0):
        qterms = set(qterms)
        res = []
        for i, c in enumerate(self.chunks):
            if c["id"] in exclude or not self.allowed(c, base):
                continue
            s = self.score(i, qterms)
            if c["kind"] == "pair":
                s *= pair_boost
            if s > 0:
                res.append((s, c["id"], i))
        res.sort(key=lambda r: (-r[0], r[1]))
        return [(self.chunks[i], round(s, 3)) for s, _, i in res[:k]]

    def distinctive(self, text, n=WIP_TERMS):
        seen = {}
        for t in tokens(text):
            if t in self.idf:
                seen[t] = self.idf[t]
        return [t for t, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

    def query1(self, task_text, wip_text, base, k=Q1_K):
        return self.search(tokens(task_text) + self.distinctive(wip_text), base, k)

    def query2(self, feedback, base, exclude=(), k=Q2_K):
        return self.search(tokens(feedback), base, k, exclude=exclude, pair_boost=PAIR_BOOST)


def render(hits):
    out = []
    for c, _ in hits:
        label = c.get("heading") or c["source"]
        out.append(f"## [{c['id']}] {label}\n\n{c['text'].strip()}\n")
    return "\n".join(out)


_INDEX = None


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = Index()
    return _INDEX
