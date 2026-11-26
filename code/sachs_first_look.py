"""First look at Sachs: log-transform, fit both estimators, compare to ground truth skeleton."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from solvers import neighbourhood_selection, glasso_support

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "data", "sachs_continuous.txt"), sep="\t")
nodes = list(df.columns)
X = np.log(df.values)                       # measurements are right-skewed; log is standard
X = (X - X.mean(0)) / X.std(0)              # standardise

# ground-truth undirected skeleton from the directed edge list
gt = np.zeros((len(nodes), len(nodes)), bool)
for line in open(os.path.join(HERE, "..", "data", "sachs_ground_truth.txt")):
    if "-->" in line:
        a, b = [s.strip() for s in line.split(".", 1)[1].split("-->")]
        i, j = nodes.index(a), nodes.index(b)
        gt[i, j] = gt[j, i] = True
print("true undirected edges:", gt.sum() // 2)

def f1(est, truth):
    iu = np.triu_indices(len(nodes), 1)
    e, t = est[iu], truth[iu]
    tp = np.sum(e & t); fp = np.sum(e & ~t); fn = np.sum(~e & t)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12), tp, fp

for lam in [0.05, 0.1, 0.2]:
    ns = neighbourhood_selection(X, lam, rule="or")
    gl, _ = glasso_support(X, lam)
    for name, est in [("neigh-or", ns), ("glasso", gl)]:
        score, tp, fp = f1(est, gt)
        print(f"lam={lam:<5} {name:<9} edges={est.sum()//2:<3} TP={tp:<3} FP={fp:<3} F1={score:.2f}")
