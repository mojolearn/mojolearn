#!/usr/bin/env python3
"""Independent scalar NumPy oracle for the deterministic UMAP schedule."""

import numpy as np

np.seterr(over="ignore")  # uint64 wraparound is the specified RNG arithmetic

A, B = np.float32(1.57694346), np.float32(0.89506088)
emb = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
w = np.array([[0, 1, .25, 1], [1, 0, 1, .25],
              [.25, 1, 0, 1], [1, .25, 1, 0]], dtype=np.float32)

def mix(x):
    x = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))

ordinal = 0
for epoch in range(5):
    alpha = np.float32((5 - epoch) / 5)
    ordinal = 0
    for head in range(4):
        for tail in range(4):
            weight = w[head, tail]
            if head == tail or not weight > 0:
                continue
            scaled = float(weight / w.max())
            if int((epoch + 1) * scaled) <= int(epoch * scaled):
                ordinal += 1
                continue
            delta = emb[head] - emb[tail]
            dist = np.float32(delta @ delta)
            if dist > 0:
                dp = np.float32(float(dist) ** float(B))
                coeff = np.float32(-2) * A * B * (dp / dist) / (A * dp + 1)
                update = alpha * np.clip(coeff * delta, -4, 4).astype(np.float32)
                emb[head] += update
                emb[tail] -= update
            for negative in range(5):
                counter = (np.uint64(23)
                           ^ np.uint64(epoch) * np.uint64(0xD1B54A32D192ED03)
                           ^ np.uint64(ordinal) * np.uint64(0x94D049BB133111EB)
                           ^ np.uint64(negative))
                other = int(mix(counter) % np.uint64(4))
                if other == head:
                    continue
                delta = emb[head] - emb[other]
                dist = np.float32(delta @ delta)
                if dist > 0:
                    dp = np.float32(float(dist) ** float(B))
                    coeff = np.float32(2) * B / ((np.float32(.001) + dist) * (A * dp + 1))
                    emb[head] += alpha * np.clip(coeff * delta, -4, 4).astype(np.float32)
            ordinal += 1
np.set_printoptions(precision=9)
print(emb)
