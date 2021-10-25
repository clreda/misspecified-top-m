#coding: utf-8

import numpy as np
import argparse
import pandas as pd
import subprocess as sb

import misspecified
from misspecified import methods
from data import create_data, data_types
import problems
from problems import problem_types
import utils
from utils import beta_types, gain_types, tracking_types

import random
from joblib import Parallel, delayed
import json

# For replication
__seed__=123456
random.seed(__seed__)
np.random.seed(__seed__)

result_folder = "../results/"
boxplot_folder = "../boxplots/"

parser = argparse.ArgumentParser(description='Models for dealing with model misspecification in linear non-contextual bandits for pure exploration')

## Common to all bandit methods
parser.add_argument('--method', type=str, choices=methods, help='Name of method.')
parser.add_argument('--data_type', type=str, choices=data_types, help='Type of feature matrix.')
parser.add_argument('--problem_type', type=str, choices=problem_types, help='Reward distribution.', default="gaussian")
parser.add_argument('--beta_linear', type=str, choices=beta_types, help='Threshold function for confidence bounds.', default="heuristic")
parser.add_argument('--delta', type=float, help='Error rate.', default=0.05)
parser.add_argument('--nsimu', type=int, help='Number of simulations.', default=500)
parser.add_argument('--omega', type=str, help='Angle in the "classic" problem.', default="pi/3")
parser.add_argument('--K', type=int, help='Number of arms.', default=3)
parser.add_argument('--N', type=int, help='Dimension of feature vectors.', default=10)
parser.add_argument('--m', type=int, help='Number of arms returned in the problem (m=1 is BAI, m>1 is Top-m).', default=1)

## Tune misspecification
parser.add_argument('--c', type=float, help='Deviation to linearity bound (in algorithm, default in data; c=0 is a linear model).', default=2)
parser.add_argument('--c_data', type=float, help='Deviation to linearity bound in data.', default=-1)
parser.add_argument('--sigma', type=float, help='Standard deviation for noise model (in algorithm, default in problem).', default=1)
parser.add_argument('--sigma_problem', type=float, help='Standard deviation for noise model in problem.', default=-1)
parser.add_argument('--epsilon', type=float, help="Slack value for stopping rule. Typically of the order of the gap between m-th and m+1-th best arm.", default=0.)
parser.add_argument('--n', type=int, help='Number of samplings per arm for uniform sampling methods.', default=2)

## Saving and running iterations
parser.add_argument('--n_jobs', type=int, help='Number of parallel processes.', default=1)
parser.add_argument('--boxplot', type=str, help="Plot boxplot associated with experiment.", default="")
parser.add_argument('--exp_name', type=str, help="Experiment name.", default="")

## Options for our method
parser.add_argument('--tracking_type', type=str, help="Type of tracking rule (if none, default of the algorithm will be used).", default="", choices=[""]+tracking_types)
parser.add_argument('--gain_type', type=str, help="Type of optimistic gains (if none, default of the algorithm will be used).", default="", choices=[""]+gain_types)
parser.add_argument('--multi_learners', type=int, help="Use one or several learners in our algorithm (default is a single learner).", default=0, choices=range(2))
parser.add_argument('--M', type=float, help="Boundness constant for l-inf models.", default=1.)

## Tune these parameters to use the algorithm on large datasets
parser.add_argument('--learner_name', type=str, help="Type of learner to use.", default="AdaHedge", choices=["AdaHedge", "Greedy"])
parser.add_argument('--subsample', type=int, help="Makes learning round faster by subsampling arms, if set to True.", default=0)
parser.add_argument('--geometric_factor', type=float, help="Used if subsample=1: size of the geometric grid for the stopping rule tests.", default=1.3)

args = parser.parse_args()

assert args.method
assert args.data_type

for arg in vars(args):
    if (arg != "omega"):
        if (str(type(vars(args)[arg])) == "<class 'str'>"):
            exec(arg+" = \""+vars(args)[arg]+"\"")
        else:
            exec(arg+" = "+str(vars(args)[arg]))

if (sigma_problem < 0):
    sigma_problem = sigma
    args.sigma_problem = sigma
if (c_data < 0):
    c_data = c
    args.c_data = c

if (len(args.exp_name) == 0):
    algo_args = ["n_jobs", "boxplot", "method", "beta_linear", "nsimu", "delta", "c", "sigma"]
    result_fname = "_".join([k+"="+str(vars(args)[k]) for k in vars(args) if (k not in algo_args)])
else:
    result_fname = exp_name
result_fname = "-".join(result_fname.split("/"))+"/"
sb.call("mkdir -p "+result_folder+result_fname, shell=True)

if (len(args.boxplot) > 0):
    sb.call("mkdir -p "+boxplot_folder+result_fname, shell=True)
    result_fname = result_folder+result_fname
    if (args.boxplot == "None"):
        methods = None
    else:
        methods = args.boxplot.split(",")
    utils.boxplot_experiment(args, result_fname, methods=methods)
    exit()

with open(result_folder+result_fname+'params.json', "w") as f:
    json.dump(vars(args), f)

result_fname = result_folder+result_fname

beta_classical=beta_linear if ("heuristic" in beta_linear) else "lucb1"
omega=eval("np.pi".join(args.omega.split("pi"))) if ("pi" in args.omega) else float(args.omega)

data_args={
        "omega": omega, 
        "K": K, "N": N, 
        "c": c_data, 
        "m": m, 
        "dr_folder": "../DR_Data/", 
        "M": M
        }

problem_args={
        "name": problem_type, 
        "sigma": sigma_problem
        }

X, scores, theta, names, pargs, M =create_data(data_type, data_args)
N, K = X.shape

mu_m, mu_m1 = utils.max_m(scores, m+1)[-2:]
score_order = np.sort(scores).tolist()
score_order.reverse()
# Prints 10 first gaps
#print([score_order[i]-score_order[i+1] for i, s in enumerate(score_order) if (i < len(score_order)-1)][:10])
print("Gap: mu^m-mu^(m+1) = "+str(mu_m-mu_m1))

if (len(pargs) > 0):
    problem_args.update(pargs)
problem_args.update({"names": names})
problem=eval("problems."+problem_type)(X, scores, theta, problem_args)

if (K <= 50):
    score_order = np.argsort(scores).tolist()
    score_order.reverse()
    arm_lst = ["Arm #%d" % x for x in score_order]
    arm_nms = ["%s" % names[x] for x in score_order]
    print("\t".join(arm_lst))
    print("\t".join(arm_nms))
    print("\t".join([str(round(scores[x],3)) for x in score_order]))
    print(pd.DataFrame(X, index=["dim #"+str(i) for i in range(N)], columns=["Arm #"+str(i) for i in range(K)])[arm_lst])
    print("")

method_args={
        "name": method, 
        "epsilon": epsilon, 
        "sigma": sigma, 
        "delta": delta, 
        "c": c, 
        "m": m, 
        "beta_linear": eval("utils."+beta_linear)(X, theta, delta, sigma, c, problem_type=problem.name), 
        "n": n, 
        "beta_classical": eval("utils."+beta_classical)(X, theta, delta, sigma, c), 
        "tracking_type": tracking_type if (len(tracking_type)>0) else None, 
        "gain_type": gain_type if (len(gain_type)>0) else None, 
        "multi_learners": (multi_learners > 0),
        "M": M,
        "subsample": bool(subsample),
        "geometric_factor": geometric_factor,
        "learner_name": learner_name,
        }
misspecified=eval("misspecified."+method)(method_args)

if args.n_jobs == 1:  # No parallelization in this case
    active_arms, complexity, regret, linearity, running_time =misspecified.run(problem, nsimu)
else:
    # Generate one random seed for each run (best practice with joblib)
    seeds = [np.random.randint(1000000) for _ in range(args.nsimu)]

    # Function to perform a single experiment
    def single_run(id, seed):
        # joblib replicates the current process, so we need to manually set a different seed for each run
        random.seed(seed)
        np.random.seed(seed)
        return misspecified.run(problem, nsimu=1, run_id=id)

    # run nsimu simulations over n_jobs parallel processes
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(delayed(single_run)(id, seed) for id, seed in enumerate(seeds))

    # we finally merge the results so as to have them in the same form as for a single process
    active_arms = np.mean(np.array([r[0] for r in results]), axis=0).tolist()
    complexity = [r[1][0] for r in results]
    regret = [r[2][0] for r in results]
    linearity = [r[3][0] for r in results]
    running_time = [r[4][0] for r in results]

# saving results 
# (sample complexity, regret, frequency of detected linearity in data, empirical recommendation on arms)
res = pd.DataFrame([], index=range(args.nsimu))
res["complexity"] = complexity
res["regret"] = regret
res["linearity"] = linearity
res["running time"] = running_time
fname = "_".join([k+"="+str(vars(args)[k]) for k in ["method", "beta_linear", "delta", "c", "sigma"]])
res.to_csv(result_fname+fname+".csv")
empirical_rec = pd.DataFrame(active_arms, columns=["freq. in rec"], index=range(K)).T
empirical_rec.to_csv(result_fname+fname+"-emp_rec.csv")

# Print summary of results
print("")
print("\t".join(["method="+method, "C="+str(np.mean(complexity)), "R="+str(np.mean(regret)), "L="+str(np.mean(linearity)), "t="+str(round(np.mean(running_time),2))+" sec."]))
print(pd.DataFrame(active_arms, columns=["Empirical rec."], index=["Arm #%d" % k for k in range(len(active_arms))]).T)
gap = lambda x : x[-2]-x[-1]
print("\t".join(["max mean="+str(round(np.max(scores),3)), "min mean="+str(round(np.min(scores),3)), "gap="+str(round(gap(utils.max_m(scores, m+1)),3))]))
