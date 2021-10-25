# coding:utf-8

import pandas as pd
import numpy as np

folder="../results/ExperimentD_tricks/"

## Produce Table in experiments (Appendix)
tps = ["AdaHedge", "Greedy", "default"]
res_di = {}
for ti, t in enumerate(tps):
    res = pd.read_csv(folder+"method=MisLid("+t+")_beta_linear=heuristic_delta=0.05_c=0.02_sigma=1.0.csv", index_col=0)
    C, _, _, T = np.round(res.mean(axis=0).values,2).flatten().tolist()
    sC, _, _, sT = np.sqrt(np.round(res.var(axis=0).values,2)).flatten().tolist()
    res_di.setdefault(t, {"avg. runtime in sec.": "%.2f (± %2.f)" % (T,sT), "avg. complexity": "%.2f (± %2.f)" % (C, sC)})

print(pd.DataFrame(res_di))
