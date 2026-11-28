Data: Sachs et al., N = 7466 cells, p = 11 proteins: raf, mek, plc, pip2, pip3, erk, akt, pka, pkc, p38, jnk
Preprocessing: log-transform each protein, then standardise each to mean 0, variance 1.

Step 1. Choosing lambda by simulation: p = 11, N = 7466; networks ER (pi = 0.3) and chain; Q by DD and PO (lambda_min = 0.1); 10 data sets per setting.
Mean F1 of the estimated network (rows: lambda = r * lambda_max):
| r | ER DD glasso | ER DD AND | ER PO glasso | ER PO AND | chain DD glasso | chain DD AND | chain PO glasso | chain PO AND | mean glasso | mean AND |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.800 | 0.706 | 0.429 | 0.429 | 0.800 | 0.750 | 0.698 | 0.571 | 0.682 | 0.614 |
| 0.2 | 0.827 | 0.900 | 0.797 | 0.778 | 0.783 | 0.947 | 0.811 | 0.889 | 0.805 | 0.879 |
| 0.1 | 0.825 | 0.952 | 0.896 | 0.900 | 0.791 | 0.947 | 0.870 | 0.947 | 0.846 | 0.937 |
| 0.05 | 0.788 | 0.962 | 0.883 | 0.900 | 0.721 | 0.968 | 0.828 | 0.974 | 0.805 | 0.951 |
| 0.02 | 0.693 | 0.962 | 0.785 | 0.881 | 0.673 | 0.986 | 0.684 | 0.937 | 0.709 | 0.941 |

Chosen lambda (best mean F1); lambda_max of the Sachs data = 0.7849:
  graphical lasso: r = 0.1, lambda = 0.0785, mean F1 in simulation = 0.846
  neighbourhood selection (AND): r = 0.05, lambda = 0.0392, mean F1 in simulation = 0.951

Network estimated on the Sachs data with the chosen lambda:
| estimator | lambda | edges | edges also in the literature network (20) |
|---|---|---|---|
| graphical lasso | 0.0785 | 31 | 13 |
| neighbourhood selection (AND) | 0.0392 | 26 | 12 |

Step 2. Stability: 50 random halves of 3733 cells, same lambda.

Step 3. Stable edges. Threshold from the gap rule: the widest gap in the selection frequencies above 0.5 is between 0.56 and 0.90, so the threshold is 0.73.
Edges passing each threshold (robustness check):
| threshold | graphical lasso | neighbourhood selection (AND) |
|---|---|---|
| 0.50 | 32 | 26 |
| 0.60 | 30 | 26 |
| 0.70 | 30 | 26 |
| 0.80 | 30 | 26 |
| 0.90 | 30 | 26 |
| 0.95 | 29 | 24 |
| 1.00 | 28 | 24 |

graphical lasso: 30 stable edges
| edge | selection frequency | in literature network |
|---|---|---|
| raf - pka | 100% | yes |
| raf - mek | 100% | yes |
| raf - erk | 100% | no |
| plc - pka | 100% | no |
| plc - pip2 | 100% | yes |
| plc - p38 | 100% | no |
| plc - jnk | 100% | no |
| plc - akt | 100% | no |
| pkc - p38 | 100% | yes |
| pkc - jnk | 100% | yes |
| pka - p38 | 100% | yes |
| pip3 - pka | 100% | no |
| pip3 - jnk | 100% | no |
| pip3 - akt | 100% | yes |
| pip2 - pka | 100% | no |
| pip2 - pip3 | 100% | yes |
| p38 - jnk | 100% | no |
| mek - pka | 100% | yes |
| mek - p38 | 100% | no |
| mek - jnk | 100% | no |
| mek - erk | 100% | yes |
| mek - akt | 100% | no |
| erk - pkc | 100% | no |
| erk - pka | 100% | yes |
| erk - jnk | 100% | no |
| erk - akt | 100% | yes |
| akt - p38 | 100% | no |
| akt - jnk | 100% | no |
| pka - jnk | 96% | yes |
| raf - akt | 94% | no |

neighbourhood selection (AND): 26 stable edges
| edge | selection frequency | in literature network |
|---|---|---|
| raf - pkc | 100% | yes |
| raf - pka | 100% | yes |
| raf - mek | 100% | yes |
| raf - jnk | 100% | no |
| plc - pka | 100% | no |
| plc - pip2 | 100% | yes |
| plc - p38 | 100% | no |
| plc - jnk | 100% | no |
| plc - akt | 100% | no |
| pkc - p38 | 100% | yes |
| pkc - jnk | 100% | yes |
| pka - p38 | 100% | yes |
| pip3 - pka | 100% | no |
| pip3 - jnk | 100% | no |
| pip2 - pip3 | 100% | yes |
| p38 - jnk | 100% | no |
| mek - jnk | 100% | no |
| mek - erk | 100% | yes |
| mek - akt | 100% | no |
| erk - pkc | 100% | no |
| erk - pka | 100% | yes |
| erk - jnk | 100% | no |
| erk - akt | 100% | yes |
| akt - p38 | 100% | no |
| pip2 - pka | 92% | no |
| pka - jnk | 90% | yes |

Consensus (high-confidence core): 24 edges stable under both estimators, 11 of them in the literature network.
| edge | freq. glasso | freq. AND | in literature network |
|---|---|---|---|
| raf - mek | 100% | 100% | yes |
| raf - pka | 100% | 100% | yes |
| mek - erk | 100% | 100% | yes |
| mek - akt | 100% | 100% | no |
| mek - jnk | 100% | 100% | no |
| plc - pip2 | 100% | 100% | yes |
| plc - akt | 100% | 100% | no |
| plc - pka | 100% | 100% | no |
| plc - p38 | 100% | 100% | no |
| plc - jnk | 100% | 100% | no |
| pip2 - pip3 | 100% | 100% | yes |
| pip2 - pka | 100% | 92% | no |
| pip3 - pka | 100% | 100% | no |
| pip3 - jnk | 100% | 100% | no |
| erk - akt | 100% | 100% | yes |
| erk - pka | 100% | 100% | yes |
| erk - pkc | 100% | 100% | no |
| erk - jnk | 100% | 100% | no |
| akt - p38 | 100% | 100% | no |
| pka - p38 | 100% | 100% | yes |
| pka - jnk | 96% | 90% | yes |
| pkc - p38 | 100% | 100% | yes |
| pkc - jnk | 100% | 100% | yes |
| p38 - jnk | 100% | 100% | no |
Stable only under graphical lasso (6): raf-erk, raf-akt, mek-pka, mek-p38, pip3-akt, akt-jnk
Stable only under neighbourhood selection (AND) (2): raf-pkc, raf-jnk

Step 4. Trust: the stable network with its estimated edge strengths is the truth; 20 data sets with N = 7466, p = 11; same lambda.
| estimator | stable edges | truth Q | TPR | FPR |
|---|---|---|---|---|
| graphical lasso | 30 | estimated strengths | 0.81 ± 0.02 | 0.140 ± 0.046 |
| neighbourhood selection (AND) | 26 | estimated strengths | 0.81 ± 0.04 | 0.003 ± 0.015 |
