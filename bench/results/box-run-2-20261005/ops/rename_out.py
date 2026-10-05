"""Rename a function parameter named `out` (reserved in this Mojo) to `dst` inside that one function. usage: file:line ..."""
import re, sys
for spec in sys.argv[1:]:
    f, ln = spec.rsplit(':', 1); ln = int(ln) - 1
    L = open(f).read().split('\n')
    d = ln
    while not re.match(r'^\s*def ', L[d]): d -= 1
    ind = len(L[d]) - len(L[d].lstrip())
    depth = 0; s = d
    while True:
        depth += L[s].count('(') + L[s].count('[') - L[s].count(')') - L[s].count(']')
        if depth <= 0 and L[s].rstrip().endswith(':'): break
        s += 1
    e = s + 1
    while e < len(L) and (not L[e].strip() or len(L[e]) - len(L[e].lstrip()) > ind): e += 1
    n = 0
    for i in range(d, e):
        new = re.sub(r'(?<![\w.])out(?!\w)', 'dst', L[i]); n += new != L[i]; L[i] = new
    open(f, 'w').write('\n'.join(L))
    print(spec, 'def', d + 1, 'end', e, 'lines changed', n)
