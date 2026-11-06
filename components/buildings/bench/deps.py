"""Couplings of the "buildings" component — so that it does not grow into a monolith. Checks the sources (ast):
  1. the core (base, siting, style, space, influences) does not call the types and ensembles; inside the
     core — no cycles;
  2. a type does not call another type (what two types share goes to the core);
  3. an ensemble calls the types only through the buildings.py entry;
  4. couplings are imports of the packages components.buildings.<part>; the old declaration
     arch_parts.need(__file__, …), if it remains, must be honest: every foreign part whose module is called
     is named (and no extra ones);
  5. knowledge (*.yaml) is read only by its own part — the file lies in the module's folder;
  6. module names are not repeated across parts (the module name is the key of this check);
and shows who calls the component from outside (info: an import of components.buildings… or a path to a
module, -m — in core/, bench/, components/, formats/, research/, projects/) and the modules longer than 1500
lines.

.venv/bin/python -m components.buildings.bench.deps   → OK/FAIL lines, __RESULT__, exit code = number of violations"""
import ast
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); COMP = os.path.dirname(HERE)
from components.buildings import arch_parts as AP                                                        # noqa: E402

ROOT = AP.ROOT
SKIP_DIRS = {"runs", "bench", "ext", "out", "__pycache__", "research"}
BIG = 1500


def modules():
    """Modules of the parts: name → (part, path); runs, references and downloads are not parts."""
    M = {}; dup = []
    for part in AP.PARTS:
        d = os.path.join(COMP, part)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and f != "__init__.py":                       # __init__ of each package is not a module of a part
                name = f[:-3]
                if name in M: dup.append((name, M[name][0], part))
                M[name] = (part, os.path.join(d, f))
    return M, dup


def parse(path):
    src = open(path, encoding="utf-8").read(); T = ast.parse(src)
    imps = set(); need = None
    PKG = "components.buildings."
    for n in ast.walk(T):
        if isinstance(n, ast.Import):
            for a in n.names:                                              # import components.buildings.<part>.<module>
                if a.name.startswith(PKG) and a.name.count(".") >= 3: imps.add(a.name.split(".")[3])
                elif not a.name.startswith(("components.", "core.", "formats.")): imps.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            if n.module.startswith(PKG):
                segs = n.module.split(".")
                if len(segs) >= 4: imps.add(segs[3])                        # from components.buildings.<part>.<module> import …
                else: imps |= {a.name for a in n.names}                    # from components.buildings.<part> import <module>
            elif not n.module.startswith(("components.", "core.", "formats.")): imps.add(n.module.split(".")[0])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "need" and getattr(n.func.value, "id", "") == "arch_parts":
            need = [a.value for a in n.args[1:] if isinstance(a, ast.Constant)]
    yamls = set(re.findall(r"[\"']([\w./-]+\.yaml)[\"']", src))
    return imps, need, yamls, src.count("\n") + 1


def main():
    M, dup = modules(); bad = []; rep = []

    def check(what, ok, val=""):
        print(("OK   " if ok else "FAIL ") + what, val if val != "" else "", flush=True)
        if not ok: bad.append(what)

    check("module names are not repeated across parts", not dup, dup or len(M))
    edges = defaultdict(set); big = []
    for name, (part, path) in sorted(M.items()):
        imps, need, yamls, n = parse(path)
        if n > BIG: big.append(f"{part}/{name}.py: {n}")
        used = {M[i][0] for i in imps if i in M and M[i][0] != part}
        for p in used: edges[part].add(p)
        if part in AP.CORE:
            wrong = sorted(p for p in used if p not in AP.CORE)
            check(f"{part}/{name}: the core does not call types and ensembles", not wrong, wrong)
        elif part in AP.TYPES:
            wrong = sorted(p for p in used if p in AP.TYPES or p in AP.ENSEMBLES)
            check(f"{part}/{name}: a type does not call another type", not wrong, wrong)
        else:
            wrong = sorted(i for i in imps if i in M and M[i][0] in AP.TYPES)
            check(f"{part}/{name}: an ensemble calls types only through the entry", not wrong, wrong)
        if need is not None:                                               # the need declaration (old) is checked only if it remains
            nd = set(need or [])
            check(f"{part}/{name}: the need declaration is honest", used <= nd and nd <= set(AP.PARTS) and not (nd - used - {part}),
                  {"calls": sorted(used), "declared": sorted(nd)})
        for y in yamls:
            base = os.path.basename(y)
            if base.endswith(".yaml") and not y.startswith(("library/", "scenes/")) and "/" not in y:
                check(f"{part}/{name}: the knowledge file {base} belongs to its own part", os.path.exists(os.path.join(os.path.dirname(path), base)), base)
    # cycles in the core
    seen, stack, cyc = set(), [], []

    def dfs(u):
        seen.add(u); stack.append(u)
        for v in edges.get(u, ()):
            if v in stack: cyc.append(stack[stack.index(v):] + [v])
            elif v not in seen: dfs(v)
        stack.pop()
    for p in AP.CORE:
        if p not in seen: dfs(p)
    check("core without cycles", not cyc, cyc or {p: sorted(edges.get(p, ())) for p in AP.CORE})
    # who calls from outside (info, not a violation): an import of the package
    # components.buildings[.<part>[.<module>]] and a reference to a module by path or -m
    # (components/buildings/castles/bl_castle.py, -m components.buildings.layers — a subprocess, a script)
    ROOTMODS = {"buildings", "layers", "steps", "arch_parts"}                 # root modules: entry, layers, step, parts
    imp = re.compile(r"(?:^|;)[ \t]*(?:from[ \t]+components\.buildings(?:\.(?P<part>\w+))?(?:\.(?P<mod>\w+))?[ \t]+import[ \t]+"
                     r"(?:\((?P<pnames>[^)]*)\)|(?P<names>[\w \t,]+))|import[ \t]+components\.buildings\.(?P<ipart>\w+)(?:\.(?P<imod>\w+))?)", re.M)
    ref = re.compile(r"components[./]buildings[./](\w+)(?:[./](\w+))?")
    outside = defaultdict(set)

    def note(nm, rel):
        if nm in M or nm in ROOTMODS: outside[nm].add(rel)
    for top in ("core", "bench", "components", "formats", "research", "projects"):
        for dp, dn, fn in os.walk(os.path.join(ROOT, top)):
            dn[:] = [d for d in dn if d not in ("ext", "out", "__pycache__", "src", ".venv", "node_modules", "site-packages") and not d.startswith(".")]
            if dp.startswith(COMP): continue
            for f in fn:
                if not f.endswith((".py", ".sh")): continue
                p = os.path.join(dp, f)
                try: s = open(p, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError): continue
                rel = os.path.relpath(p, ROOT)
                spans = []
                for m in imp.finditer(s):                                      # from components.buildings[.part[.module]] import …
                    spans.append(m.span()); g = m.groupdict(); part, mod, ipart, imod = g["part"], g["mod"], g["ipart"], g["imod"]
                    if ipart: note(imod if ipart in AP.PARTS and imod else ipart, rel)   # import components.buildings.<…>
                    elif mod: note(mod, rel)
                    elif part and part not in AP.PARTS: note(part, rel)         # from components.buildings.layers import write
                    else:
                        for a in (g["pnames"] or g["names"] or "").split(","):  # from components.buildings[.part] import module
                            if a.split(): note(a.split()[0], rel)
                for m in ref.finditer(s):                                      # a path to a module or -m: a subprocess, a script
                    if any(a <= m.start() < b for a, b in spans): continue
                    part, mod = m.groups()
                    note(mod if part in AP.PARTS and mod else part, rel + " (path)")
    rep = {k: sorted(v) for k, v in sorted(outside.items())}
    print("OUTSIDE  ", json.dumps(rep, ensure_ascii=False))
    if big: print("LARGE  ", big, "— candidates for splitting into steps")
    print("__RESULT__", json.dumps({"n_modules": len(M), "n_violations": len(bad), "links": {k: sorted(v) for k, v in edges.items()},
                                    "outside": len(rep), "n_large": len(big)}, ensure_ascii=False))
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
