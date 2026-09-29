# F1-pv32 inference on two held-out texts (job nvc2-0010, 2x RTX 4090)

F1-pv32: 15-bit codes on both operands of every projection, of the LM head and of Q.K^T; P.V in fp32.v1.
200 non-overlapping windows of 512 ids per text, 102,200 scored positions. Change is relative perplexity against arm a (fp32.v1); the interval is 95% over the window means, through exp(x) - 1. Floor: the same float32 operands accumulated in float64.

| Text | Arm | Perplexity | Change | Interval | Top-1 agreement |
|---|---|---|---|---|---|
| enwik8 | a | 7.937106 | +0.0000% | +0.0000% to +0.0000% | 1.0000 |
| enwik8 | floor | 7.937106 | +0.0000% | -0.0000% to +0.0000% | 1.0000 |
| enwik8 | F1-pv32 | 7.936947 | -0.0020% | -0.0067% to +0.0027% | 0.9979 |

| pile_github | a | 3.361991 | +0.0000% | +0.0000% to +0.0000% | 1.0000 |
| pile_github | floor | 3.361991 | +0.0000% | -0.0000% to +0.0000% | 1.0000 |
| pile_github | F1-pv32 | 3.362105 | +0.0034% | -0.0010% to +0.0078% | 0.9989 |

