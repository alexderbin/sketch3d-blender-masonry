"""The form language of buildings as a set of genes (style/styles.yaml → families):
a family with inheritance (base), a mix of families (style=a+b or a:0.6,b:0.4 — numbers are weighted, variants — weights
add up, colors blend), a choice of categorical genes by seed, gene overrides (arch=ogee, tower.spire=horns).
Used by temples/temple_story.py and temples/temple_plan.py."""
import copy, os

import numpy as np
import yaml

SK = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.yaml")))   # the form language: families, lineages
CAT = set(SK["categorical"])


def _num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)


def _merge(a, b, path=""):
    """Inheritance: b over a; categorical genes and lists are replaced whole, dicts are merged."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        p = path + str(k)
        if isinstance(v, dict) and isinstance(out.get(k), dict) and p not in CAT: out[k] = _merge(out[k], v, p + ".")
        else: out[k] = copy.deepcopy(v)
    return out


def family(name):
    """A family with inheritance through base (as a chain)."""
    if name not in SK["families"]: raise KeyError(f"no family {name}: {', '.join(SK['families'])}")
    F = SK["families"][name]
    out = _merge(family(F["base"]), F) if F.get("base") else copy.deepcopy(F)
    out.pop("base", None); out["_name"] = name
    return out


def parse(spec):
    """'a' → [(a, 1)]; 'a+b' → equally; 'a:0.6,b:0.4' → by weights (normalized)."""
    if ":" in spec or "," in spec:
        items = [(x.split(":")[0].strip(), float(x.split(":")[1]) if ":" in x else 1.0) for x in spec.split(",") if x.strip()]
    else: items = [(x.strip(), 1.0) for x in spec.split("+") if x.strip()]
    s = sum(w for _, w in items); return [(n, w / s) for n, w in items]


def _blend(items, path=""):
    keys = []
    for d, _ in items:
        for k in d:
            if k not in keys: keys.append(k)
    out = {}
    for k in keys:
        vals = [(d[k], w) for d, w in items if k in d]; ws = sum(w for _, w in vals); vals = [(v, w / ws) for v, w in vals]; p = path + str(k)
        if p in CAT:                                                             # variants: the weights add up
            acc = {}
            for v, w in vals:
                opts = v if isinstance(v, dict) else {v: 1.0}; tot = sum(opts.values()) or 1.0
                for o, q in opts.items(): acc[o] = acc.get(o, 0.0) + w * q / tot
            out[k] = acc
        elif all(isinstance(v, dict) for v, _ in vals): out[k] = _blend(vals, p + ".")
        elif all(_num(v) for v, _ in vals): out[k] = float(sum(v * w for v, w in vals))
        elif all(isinstance(v, list) and len(v) == len(vals[0][0]) and all(_num(x) for x in v) for v, _ in vals):
            ints = all(isinstance(x, int) for v, _ in vals for x in v)              # counts (bays, chapels) stay integers
            out[k] = [(int(round(sum(v[i] * w for v, w in vals))) if ints else float(sum(v[i] * w for v, w in vals))) for i in range(len(vals[0][0]))]
        else: out[k] = max(vals, key=lambda vw: vw[1])[0]                         # strings and the rest — taken from the heavier one
    return out


def resolve(spec, over=None):
    """A style from a description (a family or a mix) with gene overrides {path: value}."""
    parts = parse(spec)
    S = family(parts[0][0]) if len(parts) == 1 else _blend([(family(n), w) for n, w in parts])
    S["_name"] = spec; S["_families"] = [n for n, _ in parts]; S["_weights"] = [round(w, 3) for _, w in parts]
    for path, v in (over or {}).items():
        d = S; ks = path.split(".")
        for k in ks[:-1]: d = d.setdefault(k, {})
        d[ks[-1]] = v
    return S


def choose(w, rng):
    """A variant by weights."""
    if not isinstance(w, dict): return w
    ks = list(w); p = np.array([max(float(w[k]), 0.0) for k in ks]); p = p / p.sum() if p.sum() > 0 else np.full(len(ks), 1 / len(ks))
    return ks[int(rng.choice(len(ks), p=p))]


def pick(S, rng, path=""):
    """Categorical genes → a single value by seed (the rest unchanged). → a new dict."""
    out = {}
    for k, v in S.items():
        p = path + str(k)
        if p in CAT and isinstance(v, dict): out[k] = choose(v, rng)
        elif isinstance(v, dict) and not str(k).startswith("_"): out[k] = pick(v, rng, p + ".")
        else: out[k] = copy.deepcopy(v)
    return out


def era_family(S, step):
    """The family step eras earlier (−) or later (+) along the line (Romanesque → early → mature → late Gothic); for families
    outside the line (fantasy, sci-fi) — the same family."""
    lin = S.get("lineage"); name = S["_families"][0] if S.get("_families") else S.get("_name")
    L = SK["lineages"].get(lin) if lin else None
    if not L or name not in L: return name
    return L[int(np.clip(L.index(name) + step, 0, len(L) - 1))]


def parse_over(kw):
    """Gene overrides from the command line: keys with a dot, or known genes; numbers and lists are parsed."""
    known = set(SK["families"]["gothic_fr"]) | {"plan", "central", "ziggurat"}
    out = {}
    for k, v in list(kw.items()):
        if "." in k or k in known:
            try: val = yaml.safe_load(v)
            except Exception: val = v
            out[k] = val; kw.pop(k)
    return out
