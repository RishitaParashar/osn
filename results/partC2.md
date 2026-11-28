| Method (matched edge counts) | TPR | FPR | F1 | edges | bias on true edges |
|---|---|---|---|---|---|
| glasso | 0.739 ± 0.052 | 0.007 ± 0.003 | 0.757 ± 0.036 | 56 ± 8 | -0.331 ± 0.015 |
| reweighted glasso (log-sum) | 0.870 ± 0.027 | 0.003 ± 0.003 | 0.884 ± 0.036 | 57 ± 5 | -0.051 ± 0.011 |

chain network (59 edges), DD Q with lambda_min = 0.1, N = 100, 35 runs; reweighted glasso: log-sum penalty, eps = 0.01, up to 5 MM steps from the glasso fit; its lambda is chosen by BIC, and glasso's lambda gives the closest number of edges. bias = mean over true edges of (|q_hat| - |q|), negative = shrunk toward 0.

Sensitivity to eps (same matching; glasso columns are at the matched edge count):

| eps | edges (reweighted / glasso) | F1 (reweighted / glasso) | bias (reweighted / glasso) |
|---|---|---|---|
| 0.001 | 55 / 54 | 0.896 / 0.760 | -0.053 / -0.336 |
| 0.01 | 57 / 56 | 0.884 / 0.757 | -0.051 / -0.331 |
| 0.1 | 74 / 74 | 0.780 / 0.724 | -0.078 / -0.298 |
