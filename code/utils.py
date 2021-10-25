#coding: utf-8

import numpy as np
import pandas as pd
from random import sample, choices
from functools import reduce
from glob import glob
import quadprog
import cvxpy as cp
from time import time

######################################
## Optimization problems

## minimal distance alternative in all subproblems with best arm a != i_t
def closest_alternative(problem, b, means, theta, eta, w, c, S_t, constraint="L_inf", subsample=False, alternative_arms=[]):
    d, K = problem.X.shape
    if subsample:
        all_other_arms = [a for a in range(K) if (a not in S_t)]
        random_arms = np.random.choice(all_other_arms, d, replace=False)  # the size of the sampled arms is here
        other_arms = [a for a in all_other_arms if ((a in alternative_arms) or (a in random_arms))]
    else:
        other_arms = [a for a in range(K) if (a not in S_t)]
    all_alts = np.zeros((len(S_t), len(other_arms), K+1))
    for i in range(len(S_t)):
        for a in range(len(other_arms)):
            alt = solve_alternative_quadprog(problem, b, theta, eta, other_arms[a], S_t[i], w, c, constraint)
            all_alts[i, a, :-1] = (np.array(alt[0])).flatten()
            all_alts[i, a, -1] = alt[1]
    closest_id = np.unravel_index(all_alts[:, :, -1].argmin(), all_alts[:, :, -1].shape)
    means_alt = all_alts[closest_id[0], closest_id[1], :-1].reshape((K,1))
    val = all_alts[closest_id[0], closest_id[1], -1]
    closest = [S_t[closest_id[0]], other_arms[closest_id[1]]]
    
    closest_arm_in = S_t[closest_id[0]]
    closest_arm_other = other_arms[closest_id[1]]

    if (subsample):
        if len(alternative_arms) > d+len(S_t):
            alternative_arms.pop(0)
        if closest_arm_in not in alternative_arms:
            alternative_arms.append(closest_arm_in)
        if closest_arm_other not in alternative_arms:
            alternative_arms.append(closest_arm_other)

    return means_alt, closest, val, alternative_arms

def solve_alternative_quadprog(problem, b, theta_emp, eta_emp, a, i_t, w, epsilon, constraint=None):

    assert a != i_t

    d, K = problem.X.shape
    A = problem.X.T
    mu_emp = A.dot(theta_emp) + eta_emp
    u = np.zeros(K)
    u[a] = -1.
    u[i_t] = 1.

    P = np.zeros((K+d, K+d))
    q = np.zeros(K+d)
    G = np.zeros((2*K+1, K+d))
    h = np.zeros(2*K+1)

    D = np.diag(w)
    V = A.T.dot(D).dot(A)

    P[:d, :d] = V
    P[d:, d:] = D
    P[d:, :d] = D.dot(A)
    P[:d, d:] = (D.dot(A)).T

    q[:d] = np.array(A.T.dot(D).dot(mu_emp)).flatten()
    q[d:] = np.array(D.dot(mu_emp)).flatten()

    G[0, :] = np.hstack([A.T.dot(u), u.reshape((1,K))])
    G[1:K+1, d:] = np.identity(K)
    G[K+1:, d:] = -np.identity(K)

    h[1:] = epsilon + 1e-5

    theta_eta = quadprog_solve_qp(P, -q, G=G, h=h)

    mu_alt = A.dot(theta_eta[:d]) + theta_eta[d:]
    val = 0.5 * np.linalg.norm(np.sqrt(w)*np.array(mu_alt).flatten() - np.sqrt(w)*np.array(mu_emp).flatten())**2
    return mu_alt, val

#source:https://scaron.info/blog/quadratic-programming-in-python.html
# Wrapper for quadprog module solve_qp function for quadratic problems
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    # solve 0.5*x^T P x + q^T x, with Gx <= h and Ax = b
    cste = np.mean(np.abs(P))
    P /= cste
    reg_term = 1e-8
    while (np.linalg.det(P+reg_term*np.identity(len(q))) < 0):
        reg_term *= 10
    P += reg_term*np.identity(len(q))
    q /= np.sqrt(cste)
    G /= np.sqrt(cste)
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        A /= np.sqrt(cste)
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0] / np.sqrt(cste)

def projection(Vinv, b, x_hat, X, nb_pulls, c):
    N, K = X.shape
    sqD = np.diag(np.sqrt(nb_pulls))
    ## empirical means
    x_hat = np.diag([1./float(n) for n in nb_pulls]).dot(b)
    X = X.T
    A_N, x_N = sqD.dot(X), sqD.dot(x_hat)
    I = np.identity(K)
    R = I - A_N.dot(Vinv).dot(A_N.T)
    R = R.T.dot(R)

    if c > 0:
        h = np.array(float(c) * np.sqrt(nb_pulls))  
        # norm constraint on eta_N
        P = np.array(R + 1e-8*I)  
        # the identity matrix ensures that R has no negative eigenvalues (which are due to numerical errors)
        q = np.array(- R.dot(x_N)).flatten()
        res = quadprog_solve_qp(P, q, G=np.vstack((I, -I)), h=np.hstack((h, h)))
        eta_N = np.array(res).reshape((K, 1))
    else:
        eta_N = np.zeros_like(x_N)

    theta = Vinv.dot(A_N.T).dot(x_N - eta_N)
    eta = np.diag(1./np.sqrt(nb_pulls)).dot(eta_N)
    return theta.reshape((N,1)), eta.reshape((K, 1))

#' @param problem Problem instance as implemented in problems.py
#' @param candidates list of arm ids
#' @param Vinv NumPy Array: inverse of design matrix
#' @param b NumPy Array
#' @param c scale of deviation
#' @param Vinv_val pre-computed Vinv
#' @returns Vinv, b, means, theta, eta using samples from candidates
def update_misspecified(problem, candidates, Vinv, b, c, na, rewards, Vinv_val=None):
    N, K = problem.X.shape
    candidates.reverse()
    for ia, a in enumerate(candidates):
        i_a = -(ia+1)
        b[a,0] += rewards[i_a]
        x_a = problem.X[:, a]
        ## update Vinv
        if (str(Vinv_val) == "None"):
            Vinv = sherman_morrison(Vinv, x_a)
    if (str(Vinv_val) != "None"):
         Vinv = Vinv_val

    nb_pulls = np.array(na)
    if (np.min(nb_pulls) < 1e-10):
        nb_pulls = nb_pulls + 1e-10
    x_hat = np.diag([1./float(n) for n in nb_pulls]).dot(b)
    theta, eta = projection(Vinv, b, x_hat, problem.X, nb_pulls, c)
    means = np.array(problem.X.T.dot(theta) + eta).flatten()
    return Vinv, b, means, theta, eta

################################################
## Helper functions

#' @param w NumPy Array of length K
#' @param sum_w NumPy array of length K
#' @param na NumPy array of length K
#' @param t Python integer
#' @param tracking_type Python string character in ["C", "D", "S"]
#' @returns set of candidates to sample
def tracking_rule(w, sum_w, na, t, tracking_type, forced_exploration=False):
    assert str(na) != "None"
    K =  len(na)
    if (forced_exploration):
        undersampled = np.array(np.array(na) >= (np.sqrt(t)-0.5*K)*np.ones(np.array(na).shape), dtype=int)
        if (np.sum(undersampled)>0):
            w = (undersampled/np.sum(undersampled)).tolist()
    if (tracking_type == "C"):
        sampled = randf([float(na[a]-sum_w[a]) for a in range(K)], 1, np.min)
    elif (tracking_type == "D"):
        sampled = randf([float(na[a]-t*w[a]) for a in range(K)], 1, np.min)
    elif (tracking_type == "S"):
        sampled = choices(range(K), weights = w, k = 1)
    else:
        raise ValueError("Type of tracking rule not implemented.")
    return sampled

tracking_types = ["C", "D", "S"]

## Upper bound on (mu_k - Proj(mu_k(t)))^2 with high probability
#' @param direction NumPy Array
#' @param problem Problem instance as implemented in problems.py
#' @param na NumPy Array
#' @param t Python integer
#' @param Vinv NumPy matrix: inverse of design matrix
#' @param M upper bound on scores
#' @param c scale of deviation to linearity
#' @param x optional Python float
#' @param confidence_width 
#' @return confidence bound
def c_kt(direction, problem, na, t, Vinv, M, c, x=None, confidence_width=None, cnorm=None):
    if (str(cnorm)=="None"):
        cnorm = lambda x : np.max(np.abs(x)) # ||.||_inf norm
    N, K = problem.X.shape
    L = float(np.max(np.linalg.norm(problem.X, axis=0)))
    if (str(x)=="None"):
        x = 2*L
    X = 5 #number of good events in the analysis
    T_max = 4*M**2
    if (str(confidence_width)=="None"):
        T_lin = 2*np.log(X*t**2)+N*np.log(1+(t*L**2)/(x*N)) # theoretical value
    else:
        T_lin = confidence_width # simplified one
    e = np.exp(1)
    T_uns = 2*K*lambert(1/float(2*K)*np.log(e*X*t**3)+0.5*np.log(8*e*K*np.log(t)))
    Na = sum([1/float(n) for n in na])
    ckt = min(2*T_uns*Na, T_max)
    return min(ckt, 8*c**2+2*T_lin*mahalanobis(direction, Vinv)**2)

#' @param problem Problem instance as described in problems.py
#' @param Vinv Inverse of design matrix
#' @param mu estimated model NumPy Array
#' @param lambda_ alternative model NumPy Array
#' @param na NumPy Array
#' @param t Python integer
#' @param M upper bound on scores
#' @param c scale of deviation to linearity
#' @param gain_type Python string character
#' @x Python float
#' @return gradients of gains to feed to learner
def optimistic_gradient(problem, Vinv, mu, lambda_, na, t, M, c, gain_type, x=None):
    X = problem.X
    N, K = X.shape
    grads = np.zeros(K)
    if ((np.array(na) < 1e-10).any()):
        nb_pulls = np.array(na) + 1e-10
    else:
        nb_pulls = np.array(na)
    for a in range(K):
        ref_value = (mu-lambda_)[a]
        L = float(np.max(np.linalg.norm(X, axis=0)))
        confidence_width = np.log(t) # smaller confidence width than expected theoretically
        #confidence_wdith = 2*np.log(t)+N*np.log(1+(t*L**2)/N) # theoretical one
        if (gain_type == "unstructured"):
            deviation = np.sqrt(2*confidence_width/nb_pulls[a]) # Hoeffding's bounds
        elif (gain_type =="linear"):
            deviation = np.sqrt(2*confidence_width)*float(mahalanobis(X[:,a], Vinv)) # bounds in LinGame
        elif ("misspecified" in gain_type):
            deviation = np.sqrt(c_kt(problem.X[:,a], problem, [nb_pulls[a]], t, Vinv, M, c, x=x, confidence_width=confidence_width)) # bounds in paper
        elif (gain_type == "empirical"):
            deviation = 0.
        else:
            raise ValueError("Unimplemented type of gains: '"+gain_type+"'.")
        # gradient of gains wrt w_a for all a
        if (ref_value > 0):
            grads[a] = 0.5*(ref_value+deviation)**2
            if (gain_type == "aggressive_misspecified"):
                grads[a] = (ref_value**2+deviation**2)
        else:
            grads[a] = 0.5*(ref_value-deviation)**2
            if (gain_type == "aggressive_misspecified"):
                grads[a] = (ref_value**2+deviation**2)
        grads[a] = min(grads[a], confidence_width)
    return grads

gain_types = ["unstructured", "empirical", "linear", "misspecified", "aggressive_misspecified"]

#####################################
## Utils

#' @param M squared matrix of size (N, N)
#' @param x vector of size (N, 1)
#' @returns one-step iterative inversion of matrix (M^{-1}+xx^T)
def sherman_morrison(M, x):
    return M-(M.dot(x).dot(x.T.dot(M))/float(1+mahalanobis(x, M)**2))

#' @param M positive definite matrix of size (N, N)
#' @param x vector of size (N, 1)
#' @returns Mahalanobis norm of x wrt M
def mahalanobis(x, M):
    return np.sqrt(x.T.dot(M.dot(x)))

#' @param ls Python list of elements
#' @param m integer
#' @param f function taking a list as argument
#' and returning a value of the same type as the elements of input list
#' @return m random indices i1,i2,...,im of the list which satisfy for all i in i1,i2,...,im, ls[i] == f(ls)
def randf(ls, m, f):
    return sample(np.array(np.argwhere(np.array(ls) == f(ls))).flatten().tolist(), m)

#' @param ls Python list of elements with a total order natively implemented in Python
#' @param m integer
#' @returns m (not necessarily distinct) maximal values of the list
def max_m(ls, m):
    assert m <= len(ls)
    allowed = list(range(len(ls)))
    values = [None]*m
    for i in range(m):
        idx = np.argmax([ls[i] for i in allowed])
        values[i] = ls[allowed[idx]]
        del allowed[idx]
    return values

#' @param ls Python list of elements with a total order natively implemented in Python
#' @param m integer
#' @returns m distinct indices of the list which values are the m maximal ones (with multiplicity)
def argmax_m(ls, m):
    assert m <=len(ls)
    allowed = list(range(len(ls)))
    values = [None]*m
    for i in range(m):
        idx = randf([ls[i] for i in allowed], 1, np.max)[0]
        values[i] = allowed[idx]
        del allowed[idx]
    return values

#' @param X matrix of size (N, K)
#' @param C approximation value
#' @param quiet boolean
#' @returns a C-approximate barycentric spanner of the columns of X as done in [Awerbuch et al., 2008]
#' kind of slow, but the only other algorithm ([Amballa et al., 2021] to be published in AAAI) focuses on a specific set
def barycentric_spanner(X, C=1, quiet=True, precision=1e-6):
    N, K = X.shape
    assert K > 0
    det = lambda M : np.linalg.slogdet(M)[1]
    if (min(N,K)==K):
        return list(range(K)) #result from Awerbuch et al.
    # basis of set of arm features
    Fx = np.matrix(np.eye(N))
    F = [None]*N
    S = range(K)
    for a in range(N):
        other_ids = [u for u in range(N) if (u != a)]
        # replace Fx[:,a] with X[:,s], s in S
        max_det, max_det_id = -float("inf"), None
        for s in S:
            Xa = np.hstack((X[:,s], Fx[:,other_ids]))
            dXa = det(Xa)
            # keep it linearly independent
            if (dXa > max_det):
                max_det = dXa
                max_det_id = s
        Fx[:,a] = X[:,max_det_id]
        F[a] = max_det_id
    # transform basis into C-approximate barycentric spanner of size <= d
    done = False
    while (not done):
        found = False
        for s in S:
            for a in range(N):
                other_ids = [u for u in range(N) if (u != a)]
                det_Xs = det(np.hstack((X[:,s], Fx[:,other_ids]))) # |det(x, X_{-a})|
                det_Xa = det(Fx) # |det(X_a, X_{-a})|
                if ((det_Xs-C*det_Xa) > precision): # due to machine precision, might loop forever otherwise
                    Fx[:,a] = X[:,s]
                    F[a] = s
                    found = True
        done = not found
    spanner = [f for f in F if (str(f) != F)]
    if (not quiet):
        print("Spanner size d = "+str(len(spanner))+" | K = "+str(K)),
    return F

#' @param x vector
#' @param norm type of norm ||.||
#' @returns ||x||
def cnorm(x, norm="L_inf"):
    assert norm in ["L_inf", "L2", "L1"]
    if (norm == "L_inf"):
        return np.max(np.abs(x))
    elif (norm == "L2"):
        return np.linalg.norm(x)
    elif (norm == "L1"):
        return np.sum(np.abs(x))
    else:
        raise ValueError("Norm not implemented.")

#' @param x input
#' @returns Lambert's function for negative branch y=-1
def lambert(y, approx=False):
    if (approx):
        ## if y >= 1, use the upper bound on W_(y) for computational reasons
        W_ = lambda y : y + np.log(y) + min(0.5, 1/np.sqrt(y)) if (y >= 1.) else 1.
    else:
        from scipy.special import lambertw
        W_ = lambda y : np.real(-lambertw(-np.exp(-y), k=-1))
    return W_(y)

######################################
## Threshold functions

# [Abbasi-Yadkhori et al., 2011] Threshold function for linear models
def linear(X, theta, delta, sigma, lambda_=1., S=None, problem_type=None):
        N, K = np.shape(X)
        L = float(np.max(np.linalg.norm(X, axis=0)))
        S = float(np.linalg.norm(theta))
        def f(t, na, x=None, lambda_val=1.):
            return np.log(1/float(delta))+N*np.log(1+(t+1)*L**2/float((lambda_val+1)**2*N))+np.sqrt(lambda_val+1)/float(sigma)*S
        return f

# [Kalyanakrishnan et al., 2012] LUCB1 threshold
def lucb1(X, theta, delta, sigma, c, lambda_=None, S=None, problem_type=None):
    N, K = X.shape
    def f(t, na, x=None):
        return np.log(5*K*(t**4)/(4*delta))
    return f

# "Iterated log" inspired threshold [Kaufmann et al., 2015]
def heuristic(X, theta, delta, sigma, c, lambda_=None, S=None, problem_type=None):
    N, K = X.shape
    def f(t, na, x=None):
        return np.log((1+np.log(t+1))/float(delta))
    return f

# This paper's stopping rule for misspecified linear models
def misspecified(X, theta, delta, sigma, c, lambda_=None, S=None, problem_type=None):
    N, K = X.shape
    L = float(np.max(np.linalg.norm(X, axis=0)))
    e = np.exp(1)
    def f(t, na, x=None):
        if (str(x) == "None"):
            x = 2*L
        term_lin = (lambda t : 1+np.log(2/float(delta))+(1+1/np.log(2/float(delta)))*0.5*N*np.log(1+t*L**2/(x*N)*np.log(2/float(delta)))+2*c**2*t)
        term_uns = (lambda t : 2*K*lambert(1/float(2*K)*np.log(2*e/float(delta))+0.5*np.log(8*e*K*np.log(t))))
        return min(term_lin(t), term_uns(t))
    return f

beta_types = ["heuristic", "linear", "lucb1", "misspecified"]

######################################
## Visualization functions

def boxplot_experiment(args, result_fname, methods=None, ymaxlim=None):
    ## If error is raised, uncomment the following lines
    #import subprocess as sb
    #sb.call("python3 -m pip install seaborn==0.11.1", shell=True)
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    if (str(ymaxlim)=="None"):
        fshape=(10,10)
    else:
        fshape=(12,10)
    fig, ax = plt.subplots(figsize=fshape, nrows=1, ncols=1)
    fsize,markersize=35,20
    if (str(methods)=="None"):
        fnames = glob(result_fname+"method=*.csv")
        fnames = [a for a in fnames if ("-emp_rec" not in a)]
        methods = [f.split("method=")[-1].split("_")[0] for f in fnames]
        methods = [" (".join(m.split("(")) if (len(m.split("(")) > 1) else m for m in methods]
        methods = [r'$\varepsilon=$'.join(m.split("eps=")) if (len(m.split("eps="))>1) else m for m in methods]
    else:
        fnames = [None]*len(methods)
        for sm, m in enumerate(methods):
            res = [a for a in glob(result_fname+"method="+m+"*.csv") if ("-emp_rec" not in a)]
            if (len(res) == 0):
                raise ValueError("Algorithm "+method+" has not been run!")
            fnames[sm] = res[0]
    errors = [None]*len(methods)
    complexity = [None]*len(methods)
    for sf, f in enumerate(fnames):
        res = pd.read_csv(f)
        complexity[sf] = res["complexity"].values.flatten().tolist()
        errors[sf] = np.mean(res["regret"].values.flatten().tolist())
    ## Sort by alphabetical order of method names
    sort_ids = np.argsort(methods).tolist()
    if (any(["aggressive" in a for a in methods])):
        ## Sort by decreasing average complexity
        sort_ids = np.argsort([np.mean(c) for c in complexity]).tolist()
        sort_ids.reverse()
    complexity = [complexity[si] for si in sort_ids]
    errors = [errors[si] for si in sort_ids]
    labels = [r''+methods[si]+"\n$\hat{\delta}="+str(round(errors[ssi],3))+"$" for ssi, si in enumerate(sort_ids)]
    medianprops = dict(linestyle='-', linewidth=2.5, color='lightcoral')
    meanpointprops = dict(marker='.', markerfacecolor='white', alpha=0.)
    bplot = sns.boxplot(data=complexity, ax=ax, showmeans=False)
    colors = {True : "green", False : "red"}
    for i in range(len(methods)):
        mybox = bplot.artists[i]
        mybox.set_facecolor(colors[errors[i] <= args.delta])
    bplot = sns.stripplot(data=complexity, jitter=True, marker='o', alpha=0.25, color="grey")
    ax.plot(ax.get_xticks(), [np.mean(c) for c in complexity], "kD", label="means", markersize=markersize)
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels([int(ytick) for ytick in ax.get_yticks()], fontsize=fsize)
    ax.set_xticklabels(labels, rotation=27, fontsize=fsize, fontweight="bold")
    for ti, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color('red' if (errors[ti] > args.delta) else 'black')
    ax.set_ylabel("Sample complexity", fontsize=fsize)
    if (str(ymaxlim)!="None"):
        ax.set_ylim(0., ymaxlim)
    plt.legend(fontsize=fsize)
    boxplot_file = "../boxplots/".join(result_fname.split("../results/"))+"boxplot.png"
    plt.savefig(boxplot_file, bbox_inches="tight")
    print("Saved to "+boxplot_file)

######################################
## Tools for drug repurposing application

## SOURCE adapted from https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
def get_persistent_homology(seq):
    ## Keep track of time of birth and time of death for each peak
    class Peak:
        def __init__(self, startidx):
            self.born = self.left = self.right = startidx
            self.died = None

        def get_persistence(self, seq):
            return float("inf") if self.died is None else seq[self.born] - seq[self.died]
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by decreasing values
    indices = range(len(seq))
    indices = sorted(indices, key = lambda i: seq[i], reverse=True)
    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx-1] is not None)
        rgtdone = (idx < len(seq)-1 and idxtopeak[idx+1] is not None)
        il = idxtopeak[idx-1] if lftdone else None
        ir = idxtopeak[idx+1] if rgtdone else None
        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks)-1
        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il
        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir
        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir
    # This is optional convenience
    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)

#' @param df Pandas DataFrame of non-binary signatures (from same plate, same condition -> technical/biological replicates)
#' @param thres nbins in histogram
#' @return binary_sig binary (UP/DOWN 1/0 regulated  NA otherwise) signature for treated
def binarize_via_histogram(df, samples=[], thres=20, full=False, min_genes=5, bkgrnd_thres_upregulated=None, bkgrnd_thres_downregulated=None):
	df = df.dropna()
	genes = list(df.index)
	counts_, thresholds_ = np.histogram(df.values.flatten(), bins=thres)
	## We don't take into account bins with less than 5 genes (background noise)
	keep_ids = list(filter(lambda i : counts_[i] > min_genes, range(len(counts_))))
	counts, thresholds = counts_[keep_ids], thresholds_[keep_ids]
	if (any([str(t) == "None" for t in [bkgrnd_thres_upregulated, bkgrnd_thres_downregulated]])):
		## algorithm using persistent homology to find peaks and their significance
		## https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
		## returned peaks are already ranked by persistence/"significance" value
		ids = [peak.born for peak in get_persistent_homology(counts)]
		assert len(ids) > 0
		ids = sorted(ids[:2], key=lambda i : thresholds[i])
		if (len(ids) == 1):
			bkgrnd_thres_upregulated = thresholds[min(ids[0]+1, len(thresholds)-1)]
			## the +1 is important as we want to trim the genes in the corresponding bin
			bkgrnd_thres_downregulated = thresholds[min(max(ids[0]-1, 0)+1, len(thresholds)-1)]
			if (bkgrnd_thres_upregulated == bkgrnd_thres_downregulated):
				bkgrnd_thres_upregulated += np.sqrt(np.var(thresholds_))
		else:
			bkgrnd_thres_upregulated = thresholds[ids[-1]]
			## the +1 is important as we want to trim the genes in the corresponding bin
			bkgrnd_thres_downregulated = thresholds[ids[0]+1]
	assert bkgrnd_thres_downregulated < bkgrnd_thres_upregulated
	## Simple binarization
	signature = [None]*len(genes)
	for sidx, idx in enumerate(genes):
		if (np.mean(df.loc[idx].values) < bkgrnd_thres_downregulated):
			signature[sidx] = 0
		else:
			if (np.mean(df.loc[idx].values) > bkgrnd_thres_upregulated):
				signature[sidx] = 1
			else:
				signature[sidx] = np.nan
	binary_sig = pd.DataFrame(signature, index=genes, columns=["aggregated"])
	if (full):
		ids_ = [thresholds_.tolist().index(thresholds[i]) for i in ids]
		return [binary_sig, [bkgrnd_thres_downregulated, bkgrnd_thres_upregulated, thresholds_, counts_, ids_]]
	return binary_sig

#' @param observations_save_fname Python character string
#' @param nsteps Python integer
#' @param perts list of gene perturbations
#' @param genes Python character string list
#' @param experiments Python dictionary list (1 dictionary per experiment: keys: "cell", "dose", "ptype", "pert", "itime", "exprs")
#' @param verbose Python integer
#' @return integer: number of written experiments (write formatted experiments to file)
def write_observations_file(observations_save_fname, nsteps, perts, genes, experiments, verbose=0, save_path="", ignore=True, cell_spe_nsteps=3):
	assert all([all([k in list(exp.keys()) for k in ["cell", "dose", "ptype", "pert", "itime", "exprs"]]) for _, exp in enumerate(experiments)])
	## Create observations file
	obs_str = ""
	sig_str = ""
	perturbed_ko = list(filter(lambda x : "-" in perts[genes.index(x)], genes))
	perturbed_fe = list(filter(lambda x : "+" in perts[genes.index(x)], genes))
	ko_exists = len(perturbed_ko) > 0
	fe_exists = len(perturbed_fe) > 0
	cells = list(set([str(exp["cell"]) for _, exp in enumerate(experiments)]))
	chrom_path = False
	exp_id = 0
	for _, exp in enumerate(experiments):
		closed_genes = []
		if (chrom_path and ignore):
			if (len(closed_genes) == 0):
				continue
		pert = exp["pert"]
		if ("perturbed_oe" in list(exp.keys())):
			perturbed_oe_ = list(set(([pert] if ("trt_oe" in exp["ptype"]) else [])+exp["perturbed_oe"]))
		else:
			perturbed_oe_ = [pert] if ("trt_oe" in exp["ptype"]) else []
		if ("perturbed_ko" in list(exp.keys())):
			perturbed_ko_ = list(set(([pert] if ("trt_oe" not in exp["ptype"]) else [])+exp["perturbed_ko"]))
		else:
			perturbed_ko_ = [pert] if ("trt_oe" not in exp["ptype"]) else []		
		desc = "cell "+exp["cell"]+"; "+exp["itime"]+"; dose "+exp["dose"]+"; perturbagen "+pert+" ("+exp["ptype"]+")"
		obs_str += "// " + desc + "\n"
		pert_title = "KnockDown" if ("trt_oe" not in exp["ptype"]) else "OverExpression"
		if (pert_title == "OverExpression" and ko_exists):
			obs_str += "#Experiment"+str(exp_id+1)+"[0] |= $KnockDown"+str(exp_id+1)+";\n"
		if (pert_title == "KnockDown" and fe_exists):
			obs_str += "#Experiment"+str(exp_id+1)+"[0] |= $OverExpression"+str(exp_id+1)+";\n"
		obs_str += "#Experiment"+str(exp_id+1)+"[0] |= $"+pert_title+str(exp_id+1)+";\n"
		if ("Initial" in list(exp["exprs"].keys())):
			k = "Initial"
			obs_str += "#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"])+"] |= $"+str(k)+str(exp_id+1)+";\n"
		for k in list(exp["exprs"].keys()):
			if (k not in ["Final", "Initial"]):
				obs_str += "#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"])+"] |= $"+str(k)+str(exp_id+1)+";\n"
		if ("Final" in list(exp["exprs"].keys())):
			obs_str += "#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"])+"] |= $Final"+str(exp_id+1)+";\n"
			obs_str += "#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"]+1)+"] |= $Final"+str(exp_id+1)+";\n"
			obs_str += "fixpoint(#Experiment"+str(exp_id+1)+"["+str(exp["exprs"][k]["step"]+1)+"]);\n"
		obs_str += "\n"
		gene_fe = lambda g, perturbed_ : any([pt == g for pt in perturbed_])
		if (pert_title == "OverExpression"):
			actual_vals = [int(gene_fe(g, perturbed_oe_)) for g in perturbed_fe]
			fe_s = reduce(lambda x,y: x+" and"+y, [" FE("+g+") = "+str(actual_vals[sg]) for sg, g in enumerate(perturbed_fe)])
			sig_str += "$"+pert_title+str(exp_id+1)+" :=\n{"+fe_s+"\n};\n\n"
			if (ko_exists):
				actual_vals = [int((chrom_path and g in closed_genes) or gene_fe(g, perturbed_ko_)) for g in perturbed_ko]
				ko_s = reduce(lambda x,y: x+" and"+y, [" KO("+g+") = "+str(actual_vals[sg]) for sg, g in enumerate(perturbed_ko)])
				sig_str += "$KnockDown"+str(exp_id+1)+" :=\n{"+ko_s+"\n};\n\n"
		if (pert_title == "KnockDown"):
			actual_vals = [int(gene_fe(g, perturbed_ko_) or (chrom_path and g in closed_genes)) for g in perturbed_ko]
			ko_s = reduce(lambda x,y: x+" and"+y, [" KO("+g+") = "+str(actual_vals[sg]) for sg, g in enumerate(perturbed_ko)])
			sig_str += "$"+pert_title+str(exp_id+1)+" :=\n{"+ko_s+"\n};\n\n"
			if (fe_exists):
				actual_vals = [int(gene_fe(g, perturbed_oe_)) for g in perturbed_fe]
				fe_s = reduce(lambda x,y: x+" and"+y, [" FE("+g+") = "+str(actual_vals[sg]) for sg, g in enumerate(perturbed_fe)])
				sig_str += "$OverExpression"+str(exp_id+1)+" :=\n{"+fe_s+"\n};\n\n"
		for k in list(exp["exprs"].keys()):
			sig = exp["exprs"][k]["sig"]
			degs = list(sorted(sig.dropna().index))
			sig_s = reduce(lambda x,y: x+" and"+y, [" "+g+" = "+str(int(int(sig.loc[g]) > 0 and (not chrom_path or not g in closed_genes))) for g in degs])
			sig_str += "$"+str(k)+str(exp_id+1)+" :=\n{"+sig_s+"\n};\n\n"
		exp_id += 1
	with open(observations_save_fname, "wb+") as f:
		f.write((obs_str+"\n"+sig_str).encode('utf8'))
	return exp_id+1

#' @param x NumPy array of float of size n
#' @param P NumPy array of float of size n
#' @return Python float: cosine score value
def cosine_score(x, P):
	'''Computes cosine score S : x, P -> 1-(x.P)/(||x||.||P||)'''
	df = x.join(P, how="inner")
	x, P = df[[df.columns[0]]], df[[df.columns[1]]]
	cos = float(np.dot(np.transpose(x), P)/(np.linalg.norm(x, 2)*np.linalg.norm(P, 2)))
	return cos

## apply binary mask on binary vector
#' @param vector binary DF
#' @param mask binary DF
#' @return result Data Frame of the masked vector
def apply_mask(vector, mask):
	masked = vector.join(mask, how="outer")[[mask.columns[0]]]
	replaces = list(filter(lambda x : x in vector.dropna().index, masked.index[pd.isnull(masked).any(1).values.nonzero()[0]]))
	for row in replaces:
		masked.loc[row] = vector.loc[row][vector.columns[0]]
	masked.columns = ["masked"]
	return masked

#' @param output Python character string
#' @param t Python integer
#' @param genes Python list of character strings
#' @param solmax Python integer
#' @param ch Python integer
#' @return df Pandas DataFrame of state (#genes rows x 1 column)
def get_state(output, t, genes, solmax=0, ch=None):
	select_trajectory = lambda state, ch : state[(ch*len(genes)):((ch+1)*len(genes))]
	to_df = lambda state : pd.DataFrame(state, index=genes, columns=["state"])
	lines = list(filter(lambda x : "step"+str(t)+" " in x, output.split("\n")))
	state = "".join("".join("".join(lines).split("step"+str(t)+" ")).split(" "))
	state = list(map(int, state))
	if (str(ch) == "None"):
		## Select a trajectory at random
		if (solmax == 0):
			solmax = int(len(state)/float(len(genes)))
		ch = np.random.choice(range(solmax), p=[1/float(solmax)]*solmax)
		state = select_trajectory(state, ch)
		assert len(state) == len(genes)
		return to_df(state), ch
	state = select_trajectory(state, ch)
	assert len(state) == len(genes)
	return to_df(state)
