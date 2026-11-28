| Network | Method | N | Estimator | TPR @ FPR=0.05 | F1 (CV) | edges (CV) | F1 (BIC) | edges (BIC) | F1 oracle* | glasso failures |
|---|---|---|---|---|---|---|---|---|---|---|
| ER (83) | DD | 20 | glasso | 0.33 | 0.40 ± 0.03 | 112 | 0.19 ± 0.16 | 25 | 0.39 | 4 |
| ER (83) | DD | 100 | glasso | 0.69 | 0.48 ± 0.03 | 213 | 0.59 ± 0.04 | 140 | 0.66 | 0 |
| ER (83) | DD | 400 | glasso | 0.85 | 0.47 ± 0.03 | 254 | 0.61 ± 0.04 | 166 | 0.80 | 0 |
| ER (83) | DD | 20 | neigh. sel. (OR) | 0.35 | 0.40 ± 0.03 | 126 | 0.23 ± 0.01 | 461 | 0.41 | – |
| ER (83) | DD | 100 | neigh. sel. (OR) | 0.73 | 0.48 ± 0.04 | 213 | 0.63 ± 0.03 | 122 | 0.71 | – |
| ER (83) | DD | 400 | neigh. sel. (OR) | 0.89 | 0.47 ± 0.04 | 253 | 0.71 ± 0.04 | 131 | 0.87 | – |
| ER (83) | PO | 20 | glasso | 0.25 | 0.32 ± 0.05 | 79 | 0.24 ± 0.10 | 29 | 0.32 | 0 |
| ER (83) | PO | 100 | glasso | 0.48 | 0.44 ± 0.03 | 152 | 0.50 ± 0.04 | 87 | 0.51 | 0 |
| ER (83) | PO | 400 | glasso | 0.63 | 0.46 ± 0.03 | 205 | 0.59 ± 0.03 | 114 | 0.62 | 0 |
| ER (83) | PO | 20 | neigh. sel. (OR) | 0.26 | 0.32 ± 0.04 | 95 | 0.21 ± 0.01 | 449 | 0.32 | – |
| ER (83) | PO | 100 | neigh. sel. (OR) | 0.51 | 0.43 ± 0.04 | 159 | 0.53 ± 0.04 | 77 | 0.54 | – |
| ER (83) | PO | 400 | neigh. sel. (OR) | 0.68 | 0.46 ± 0.03 | 207 | 0.64 ± 0.02 | 98 | 0.69 | – |
| hub (36) | DD | 20 | glasso | 0.62 | 0.52 ± 0.07 | 49 | 0.35 ± 0.28 | 16 | 0.53 | 0 |
| hub (36) | DD | 100 | glasso | 0.88 | 0.56 ± 0.06 | 78 | 0.74 ± 0.05 | 48 | 0.84 | 0 |
| hub (36) | DD | 400 | glasso | 0.92 | 0.54 ± 0.07 | 88 | 0.76 ± 0.06 | 50 | 0.92 | 0 |
| hub (36) | DD | 20 | neigh. sel. (OR) | 0.61 | 0.47 ± 0.12 | 53 | 0.11 ± 0.01 | 458 | 0.54 | – |
| hub (36) | DD | 100 | neigh. sel. (OR) | 0.87 | 0.53 ± 0.06 | 85 | 0.77 ± 0.05 | 44 | 0.85 | – |
| hub (36) | DD | 400 | neigh. sel. (OR) | 0.91 | 0.54 ± 0.06 | 90 | 0.80 ± 0.05 | 46 | 0.92 | – |
| hub (36) | PO | 20 | glasso | 0.38 | 0.38 ± 0.08 | 30 | 0.29 ± 0.12 | 13 | 0.35 | 0 |
| hub (36) | PO | 100 | glasso | 0.65 | 0.49 ± 0.07 | 61 | 0.58 ± 0.05 | 29 | 0.58 | 0 |
| hub (36) | PO | 400 | glasso | 0.83 | 0.50 ± 0.05 | 87 | 0.70 ± 0.05 | 44 | 0.74 | 0 |
| hub (36) | PO | 20 | neigh. sel. (OR) | 0.37 | 0.36 ± 0.08 | 42 | 0.11 ± 0.01 | 437 | 0.34 | – |
| hub (36) | PO | 100 | neigh. sel. (OR) | 0.63 | 0.47 ± 0.08 | 65 | 0.58 ± 0.05 | 30 | 0.59 | – |
| hub (36) | PO | 400 | neigh. sel. (OR) | 0.83 | 0.50 ± 0.06 | 89 | 0.71 ± 0.05 | 44 | 0.75 | – |

The number in brackets after the network is its true number of edges. Edges (CV/BIC) is the mean number of edges in the selected network.
*Oracle F1: the F1 of the single best fixed λ/λ_max on the grid, averaged over runs. It uses the true network, so it is not a usable rule; it is a reference. CV and BIC choose λ separately in each run, so they can occasionally beat it.
