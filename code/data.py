# coding: utf-8

import subprocess as sb
import numpy as np
import pickle
import os
import pandas as pd
from copy import deepcopy
from glob import glob

import utils

data_types = [
        ## Linear models
        "soare", 
        ## Misspecified linear models
        "deviation_Linf", "deviation_L2", "deviation_L1", "misspecified_linear", 
        ## Unstructured
        "unstructured",
        ## Real-life models
        "epilepsy", "simple_epilepsy", "simple_epilepsy_m10", 
        ## "Linearized" models from real-life data
        "linearized_dr", "linearized_lastfm", "small_dr", "small_lastfm",
    ]

data_folder = "../data/"
grn_inference_folder = "../expansion-network/"
dr_folder = "../DR_Data/"

def linearized_pathnames(K, d, name, root_folder="./"):
    path = root_folder
    if ("_dr" in name):
        path += "problems/dr_175/representations/"
    else:
        path += "problems/lastfm/representations/"
    if ("small" in name):
        path += "small/"
    if ("lastfm" in name):
        path += "lastfm_problem2"
    pathname = path+"*_k"+str(K)+"_d"+str(d)+"_maxdev*.npz"
    files = glob(pathname)
    if (len(files) != 1):
        raise ValueError("Data not available: path: '"+pathname+"'")
    C_DATA_VALUE = float(files[0].split("_maxdev")[-1].split(".npz")[0])
    return np.load(files[0]), C_DATA_VALUE

def import_df(fname):
	df = pd.read_csv(fname, index_col=0)
	df = df.loc[~df.index.duplicated()]
	return df

#' @param eta linked to isotropic covariance in the posterior Bayesian distribution: the largest it is, the more uniform is the distribution
#' @param L upper bound on norm of feature vectors
#' @param S upper bound on norm of theta
def create_data(data_type, data_args, precision=1e-7, eta=20., L=1., S=1.):
    assert "K" in data_args and "N" in data_args
    K, N = data_args["K"], data_args["N"]
    M = data_args["M"] #upper bound on norm of bandit model
    # default is l_inf norm
    norm = "L2" if ("deviation_L2" in data_type) else ("L1" if ("deviation_L1" in data_type) else "L_inf")
    instance_indep_params = ["dr_folder"]+(["m"] if (data_type in ["misspecified_linear", "epilepsy", "simple_epilepsy", "linearized_dr", "linearized_lastfm"]) else [])
    fname = data_folder+"DATA_type="+data_type+"_"+"_".join([k+"="+str(data_args[k]) for k in data_args if (k not in instance_indep_params)])+".pck"
    ## use previous instance if already generated
    if (os.path.exists(fname)):
        print("found a saved instance with those parameters in", fname, ", loading it.")
        with open(fname, "rb") as f:
            di = pickle.load(f)
            return di["X"], di["scores"], di["theta"], di["names"], di["pargs"], utils.cnorm(di["scores"], norm=norm)
    print("creating data")
    # --------------------
    # MISSPECIFIED CLASSIC
    # --------------------
    # Linear model mu = X^T theta where mu_(m+1) is the (m+1)-th best arm
    # we build mu' such that for all i != (m+1), mu'_i = mu_i and mu'_(m+1) = mu_(m+1)+c
    if (data_type == "misspecified_linear"):
        data_args_linear = deepcopy(data_args)
        data_args_linear.update({"c": 0.})
        X, scores, theta, names, pargs, M = create_data("deviation", data_args_linear)
        assert data_args["m"] < K
        am, a = utils.argmax_m(scores, data_args["m"]+1)[-2:]
        mu_m = scores[am]
        scores[a] += data_args["c"]
        if (scores[a] > mu_m):
            print("Hard instance! mu_a'-mu_a* = "+str(scores[a]-mu_m))
        else:
            print("Easy instance: mu_a'-mu_a* = "+str(scores[a]-mu_m))
        C_DATA_VALUE = data_args["c"]
    # --------------------
    # UNSTRUCTURED
    # --------------------
    # mu = X^T theta where the columns of X are a canonical basis of R^K
    if (data_type == "unstructured"):
        X = np.matrix(np.eye(K)).reshape((K,K))
        scores = np.matrix(np.random.normal(0., eta, size=(1,K))).reshape((1, K))
        scores /= utils.cnorm(scores, norm=norm)/M
        theta = deepcopy(scores)
        names = np.array(range(K))
        pargs = {}
        C_DATA_VALUE = float("inf")
        assert utils.cnorm(scores, norm=norm)-M < precision
    # --------------------
    # DEVIATED FROM LINEAR
    # --------------------
    # mu = X^T theta + eta
    # with ||eta||_inf <= c (c = 0 => linear)
    if ("deviation" in data_type):
        c = data_args['c']
        X = np.matrix(np.random.normal(0., 5., size=(N, K))).reshape((N,K))
        X /= np.array([utils.cnorm(X[:,a], norm=norm) for a in range(K)])/L
        theta = np.matrix(np.random.normal(0., 20., size=(N, 1))).reshape((N,1))
        theta /= utils.cnorm(theta, norm=norm)/S
        scores = theta.T.dot(X)
        names = np.array(range(K))
        pargs = {}
        if (c == 0):
            C = np.matrix(np.zeros((1, K))).reshape((1,K))
        else:
            assert "c" in data_args and data_args["c"] > 0
            max_iter = 1000
            for it in range(max_iter):
                C = np.matrix(np.random.normal(0, 20., size=(1, K))).reshape((1, K))
                ## checking that C does not belong to the span of column vectors in X
                A = np.vstack((X, C))
                valid_C = (np.linalg.matrix_rank(A) >= np.linalg.matrix_rank(X))
                if (valid_C):
                    break
            if (not valid_C):
                raise ValueError("C is not a valid vector: retry...")
            C /= utils.cnorm(C, norm=norm)/c
        scores = theta.T.dot(X) + C
        assert utils.cnorm(C, norm=norm)-c < precision
        C_DATA_VALUE = c
    # ----------------------
    # Soare's pathological instance for linear BAI (extended in Réda et al. to Top-m identification)
    # ----------------------
    # mu = theta^X
    # X = [[1 1 1 1 ... 1 0 0 0, cos(omega)], (m 1's, K-1-m 0's)
    #      [0 1 0 0 ... 0 0 0 0,     0     ],
    #      [0 0 1 0 ... 0 0 0 0,     0     ],
    #      [        ...                    ],
    #      [0 0 0 0 ... 0 0 0 1, sin(omega)],
    # theta = [1 0 0 ... 0]^T
    if (data_type == "soare"):
        assert "omega" in data_args
        omega = data_args["omega"]
        assert "m" in data_args and data_args["m"] < data_args["K"] and data_args["m"] > 0
        m = data_args["m"]
        X = np.matrix(np.eye(K-1)).reshape((K-1, K-1))
        X[0,:m] = 1
        e_w = np.matrix([[np.cos(omega)]+[0]*(K-1-2)+[np.sin(omega)]]).reshape((K-1, 1))
        X = np.hstack((X, e_w))
        theta = np.matrix([[1]+[0]*(K-2)]).reshape((K-1, 1))
        scores = theta.T.dot(X)
        names = np.array(range(K))
        pargs = {}
        C_DATA_VALUE = 0
    # ----------------------
    # Drug repurposing applied to epilepsy
    # ----------------------
    ## True call to the simulator for each reward ("black-box" setting)
    if (data_type == "epilepsy"):
        path_to_grn=grn_inference_network+"examples/models/m30/"
        grn_name="report_model1_solution.net"
        pargs = {"grn_name": grn_name, "dr_folder": dr_folder, "path_to_grn": path_to_grn}
        assert "dr_folder" in data_args
        P = import_df(data_args["dr_folder"]+"GSE77578_patients.csv") ##patients
        phenotype = import_df(data_args["dr_folder"]+"epilepsy_phenotype.csv") ##CD[patients||controls]
        binarized_D = pd.DataFrame([1-int(s) for s in phenotype[phenotype.columns[0]]], index=phenotype.index, columns=["-"+phenotype.columns[0]]) ##CD[controls||patients]
        binarized_X = import_df(data_args["dr_folder"]+"epilepsy_signatures_binarized.csv") ##binarized drug signatures
        X = import_df(data_args["dr_folder"]+"epilepsy_signatures_nonbinarized.csv") ##drug signatures
        scores = import_df(data_args["dr_folder"]+"epilepsy_scores.csv")
        X = np.matrix(X.values)
        names = list(scores["drug_name"])
        scores = scores["score"].values
        theta = np.linalg.inv(X.dot(X.T)).dot(X.dot(scores).T) ## least-squares linear regression
        pargs.update({"binarized_X": binarized_X, "binarized_D": binarized_D, "P": P})
        C_DATA_VALUE = utils.cnorm(theta.T.dot(X).reshape(K)-scores.reshape(K), norm=norm)
    ## Simulator has been run and associated stochastic rewards (from 18 patients) have been saved in a file for each treatment
    ## Considering a subset of treatments of size 175 or 10
    if ("simple_epilepsy" in data_type):
        assert K in [10,175]
        X = import_df(data_args["dr_folder"]+"epilepsy_signatures_nonbinarized_"+str(K)+"drugs"+("" if (data_type == "simple_epilepsy") else "_m10")+".csv") ##drug signatures
        names = X.columns
        X = np.matrix(X.values)
        N, K = X.shape
        ## Remove non-varying features
        if (N > K):
            var_order = np.argsort(np.var(X, axis=1).T.flatten()).flatten().tolist()[0][-K:]
            X = X[var_order, :]
            N, K = X.shape
        scores = pd.read_csv(data_args["dr_folder"]+"rewards_cosine_"+str(K)+"drugs_18samples"+("" if (data_type == "simple_epilepsy") else "_m10")+".csv", header=0, index_col=0).mean(axis=0).values
        K = scores.shape[0]
        theta = np.linalg.inv(X.dot(X.T)).dot(X.dot(scores).T) ## least-square linear regression
        pargs = {}
        C_DATA_VALUE = utils.cnorm(theta.T.dot(X).reshape(K)-scores.reshape(K), norm=norm)
    if (data_type in ["linearized_dr", "small_dr"]):
        names = list(import_df(data_args["dr_folder"]+"epilepsy_signatures_nonbinarized_"+str(K)+"drugs.csv").columns)
        res, C_DATA_VALUE = linearized_pathnames(K, N, data_type)
        X = np.matrix(res["features"].T)
        theta = res["theta"].reshape((res["theta"].shape[0],1))
        scores = pd.read_csv(data_args["dr_folder"]+"rewards_cosine_"+str(K)+"drugs_18samples.csv", header=0, index_col=0).mean(axis=0).values
        if (data_type == "small_dr"):
            arm_ids = res["arm_ids"]
            scores = scores[arm_ids]
            X = X[:,arm_ids]
            names = names[arm_ids]
        pargs = {}
    if (data_type in ["linearized_lastfm", "small_lastfm"]):
        res, C_DATA_VALUE = linearized_pathnames(K, N, data_type)
        X = np.matrix(res["features"].reshape((K, N)).T)
        names = list(range(K))
        theta = res["theta"].reshape((res["theta"].shape[0],1))
        scores = res["ground_truth"]
        pargs = {}
    scores = np.array(scores).flatten().tolist()
    with open(fname, "wb") as f:
        pickle.dump({"X": X, "scores": scores, "theta": theta, "names": names, "pargs": pargs}, f)
    return X, scores, theta, names, pargs, utils.cnorm(scores, norm=norm) # = M
