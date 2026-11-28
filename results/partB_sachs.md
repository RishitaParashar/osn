Sachs data: N = 7466, p = 11, true undirected edges = 20 (Sachs skeleton)

Step 2. lambda by simulation (20 data sets, F1 against the Sachs skeleton):
  graphical lasso: best lambda/lambda_max = 0.056, mean F1 there = 0.89
  neighbourhood sel. (AND): best lambda/lambda_max = 0.026, mean F1 there = 0.95

Step 3. estimates on the Sachs data with the chosen lambda:
| estimator | lambda/lambda_max | lambda | edges | TP | FP | F1 vs skeleton |
|---|---|---|---|---|---|---|
| graphical lasso | 0.056 | 0.0441 | 36 | 14 | 22 | 0.50 |
| neighbourhood sel. (AND) | 0.026 | 0.0205 | 34 | 15 | 19 | 0.56 |

Step 4. stability: 50 random halves of 3733 samples each, same lambda.

Step 5. stable edges (selected in at least 80% of the halves):

graphical lasso: 34 stable edges
  raf - pkc: selected 100% of halves, in Sachs skeleton: yes
  raf - pka: selected 100% of halves, in Sachs skeleton: yes
  raf - mek: selected 100% of halves, in Sachs skeleton: yes
  raf - jnk: selected 100% of halves, in Sachs skeleton: no
  raf - erk: selected 100% of halves, in Sachs skeleton: no
  raf - akt: selected 100% of halves, in Sachs skeleton: no
  plc - pka: selected 100% of halves, in Sachs skeleton: no
  plc - pip2: selected 100% of halves, in Sachs skeleton: yes
  plc - p38: selected 100% of halves, in Sachs skeleton: no
  plc - jnk: selected 100% of halves, in Sachs skeleton: no
  plc - akt: selected 100% of halves, in Sachs skeleton: no
  pkc - p38: selected 100% of halves, in Sachs skeleton: yes
  pkc - jnk: selected 100% of halves, in Sachs skeleton: yes
  pka - p38: selected 100% of halves, in Sachs skeleton: yes
  pka - jnk: selected 100% of halves, in Sachs skeleton: yes
  pip3 - pka: selected 100% of halves, in Sachs skeleton: no
  pip3 - jnk: selected 100% of halves, in Sachs skeleton: no
  pip3 - akt: selected 100% of halves, in Sachs skeleton: yes
  pip2 - pka: selected 100% of halves, in Sachs skeleton: no
  pip2 - pip3: selected 100% of halves, in Sachs skeleton: yes
  p38 - jnk: selected 100% of halves, in Sachs skeleton: no
  mek - pka: selected 100% of halves, in Sachs skeleton: yes
  mek - p38: selected 100% of halves, in Sachs skeleton: no
  mek - jnk: selected 100% of halves, in Sachs skeleton: no
  mek - erk: selected 100% of halves, in Sachs skeleton: yes
  mek - akt: selected 100% of halves, in Sachs skeleton: no
  erk - pkc: selected 100% of halves, in Sachs skeleton: no
  erk - pka: selected 100% of halves, in Sachs skeleton: yes
  erk - jnk: selected 100% of halves, in Sachs skeleton: no
  erk - akt: selected 100% of halves, in Sachs skeleton: yes
  akt - p38: selected 100% of halves, in Sachs skeleton: no
  plc - erk: selected 98% of halves, in Sachs skeleton: no
  raf - pip2: selected 92% of halves, in Sachs skeleton: no
  pip3 - erk: selected 82% of halves, in Sachs skeleton: no
  pairs selected in more than half of the halves: 36

neighbourhood sel. (AND): 32 stable edges
  raf - pkc: selected 100% of halves, in Sachs skeleton: yes
  raf - pka: selected 100% of halves, in Sachs skeleton: yes
  raf - mek: selected 100% of halves, in Sachs skeleton: yes
  raf - jnk: selected 100% of halves, in Sachs skeleton: no
  plc - pka: selected 100% of halves, in Sachs skeleton: no
  plc - pip2: selected 100% of halves, in Sachs skeleton: yes
  plc - p38: selected 100% of halves, in Sachs skeleton: no
  plc - jnk: selected 100% of halves, in Sachs skeleton: no
  plc - akt: selected 100% of halves, in Sachs skeleton: no
  pkc - p38: selected 100% of halves, in Sachs skeleton: yes
  pkc - jnk: selected 100% of halves, in Sachs skeleton: yes
  pka - p38: selected 100% of halves, in Sachs skeleton: yes
  pip3 - pka: selected 100% of halves, in Sachs skeleton: no
  pip3 - jnk: selected 100% of halves, in Sachs skeleton: no
  pip2 - pka: selected 100% of halves, in Sachs skeleton: no
  pip2 - pip3: selected 100% of halves, in Sachs skeleton: yes
  p38 - jnk: selected 100% of halves, in Sachs skeleton: no
  mek - pkc: selected 100% of halves, in Sachs skeleton: yes
  mek - jnk: selected 100% of halves, in Sachs skeleton: no
  mek - erk: selected 100% of halves, in Sachs skeleton: yes
  mek - akt: selected 100% of halves, in Sachs skeleton: no
  erk - pkc: selected 100% of halves, in Sachs skeleton: no
  erk - pka: selected 100% of halves, in Sachs skeleton: yes
  erk - p38: selected 100% of halves, in Sachs skeleton: no
  erk - jnk: selected 100% of halves, in Sachs skeleton: no
  erk - akt: selected 100% of halves, in Sachs skeleton: yes
  akt - pkc: selected 100% of halves, in Sachs skeleton: no
  akt - p38: selected 100% of halves, in Sachs skeleton: no
  raf - p38: selected 98% of halves, in Sachs skeleton: no
  plc - erk: selected 98% of halves, in Sachs skeleton: no
  pka - jnk: selected 98% of halves, in Sachs skeleton: yes
  plc - pip3: selected 90% of halves, in Sachs skeleton: yes
  pairs selected in more than half of the halves: 35

Step 6. trust: simulate 20 data sets from the stable network (N = 7466, p = 11), estimate with the same relative lambda, compare with the stable network.
| estimator | stable edges (truth) | TPR | FPR | F1 |
|---|---|---|---|---|
| graphical lasso | 34 (restricted glasso estimate) | 0.83 ± 0.03 | 0.190 ± 0.044 | 0.85 |
| neighbourhood sel. (AND) | 32 (restricted glasso estimate) | 0.81 ± 0.03 | 0.026 ± 0.026 | 0.88 |
