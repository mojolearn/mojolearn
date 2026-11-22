#!/usr/bin/env python3
"""Dependency-free scalar LM oracle for UMAP curve parameters."""

import math

def fit(min_dist, spread):
    def target(x):
        return 1.0 if x < min_dist else math.exp(-(x-min_dist)/spread)
    xs = [spread * 3.0 * i / 299.0 for i in range(300)]
    def loss(a, b):
        return sum((1/(1+a*x**(2*b))-target(x))**2 for x in xs)
    a = b = 1.0
    damping = 1e-3
    current = loss(a, b)
    for _ in range(64):
        haa = hab = hbb = ga = gb = 0.0
        for x in xs[1:]:
            p = x**(2*b); den = 1+a*p
            r = 1/den-target(x)
            ja = -p/(den*den); jb = -a*p*2*math.log(x)/(den*den)
            haa += ja*ja; hab += ja*jb; hbb += jb*jb
            ga += ja*r; gb += jb*r
        haa += damping; hbb += damping
        det = haa*hbb-hab*hab
        da = (-hbb*ga+hab*gb)/det; db = (hab*ga-haa*gb)/det
        na, nb = a+da, b+db
        if na > 1e-12 and nb > 1e-12 and loss(na, nb) < current:
            a, b = na, nb; current = loss(a, b); damping *= .5
        else:
            damping *= 10
    return a, b, current

for pair in ((.1, 1.0), (.35, 1.7)):
    print(pair, fit(*pair))
