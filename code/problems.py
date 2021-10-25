#coding: utf-8

import numpy as np
import os
import utils
import subprocess as sb
import pandas as pd

problem_types=["bernouilli", "gaussian", "poisson", "exponential", "drug_repurposing", "simple_epilepsy"]

######################################

class problem(object):
    def __init__(self, X, scores, theta, problem_args):
        self.X = X
        self.scores = scores
        self.theta = theta
        for k in problem_args:
            setattr(self, k, problem_args[k])

    def reward(self, arm):
        raise NotImplemented("Top class: choose one of the class instances instead!")

######################################

class bernouilli(problem):
    def __init__(self, X, scores, theta, problem_args):
        self.name = "bernouilli"
        super(bernouilli, self).__init__(X, scores, theta, problem_args)

    def reward(self, arm):
        assert self.scores[arm] >= 0 and self.scores[arm] <= 1
        return float(np.random.binomial(1, self.scores[arm], size=1))

######################################

class gaussian(problem):
    def __init__(self, X, scores, theta, problem_args):
        self.name = "gaussian"
        assert "sigma" in problem_args and problem_args["sigma"] > 0
        super(gaussian, self).__init__(X, scores, theta, problem_args)
        
    def reward(self, arm):
        return float(np.random.normal(self.scores[arm], self.sigma, size=1))

######################################

class poisson(problem):
    def __init__(self, X, scores, theta, problem_args):
        self.name = "poisson"
        super(poisson, self).__init__(X, scores, theta, problem_args)
        
    def reward(self, arm):
        assert self.scores[arm] >= 0
        return float(np.random.poisson(self.scores[arm], size=1))

######################################

class exponential(problem):
    def __init__(self, X, scores, theta, problem_args):
        self.name = "exponential"
        super(exponential, self).__init__(X, scores, theta, problem_args)
        
    def reward(self, arm):
        assert self.scores[arm] > 0
        return float(np.random.exponential(self.scores[arm], size=1))

######################################

class drug_repurposing(problem):
    def __init__(self, X, scores, theta, problem_args):
        assert "path_to_grn" in problem_args
        assert "dr_folder" in problem_args
        assert "binarized_X" in problem_args
        assert "binarized_D" in problem_args
        assert "P" in problem_args
        assert "grn_name" in problem_args
        super(drug_repurposing, self).__init__(X, scores, theta, problem_args)
        self.name = "drug_repurposing"
        ## Store in the expansion-network program the GRN file
        if (not os.path.exists(self.path_to_grn+self.grn_name)):
            if (not os.path.exists(self.path_to_grn)):
                sb.call("mkdir "+self.path_to_grn, shell=True)
            sb.call("cp "+self.dr_folder+self.grn_name+" "+self.path_to_grn, shell=True)
        with open(self.dr_folder+self.grn_name, "r") as f:
            model = f.read()
            self.genes = [x.split("[")[0].upper() for x in model.split("\n")[5].split("; ")][:-1]
        ## Note in the model that any gene in the GRN can be perturbed
        with open(self.path_to_grn+self.grn_name, "w") as f:
            model = model.split("\n")
            buildgene = lambda x : x.split("[")[0]+"[+-]{"+x.split("{")[-1].split("}")[0]+"}("+x.split("(")[-1].split(")")[0]+")"
            model[5] = "; ".join([buildgene(x) for x in model[5].split("; ")][:-1])+"; "
            f.write("\n".join(model))
        self.perts = ["-+"]*len(self.genes)
        ## Scoring function: non-adjusted cosine score
        self.func = utils.cosine_score

    def print_state(self, state):
        ones = pd.DataFrame([1]*len(self.genes), index=self.genes, columns=["ones"])
        df = state.join(ones, how="outer").fillna(-1)
        df = df[[df.columns[0]]]
        df[df.columns[0]] = np.asarray(np.asarray(df[[df.columns[0]]].values, dtype=int), dtype=str)
        df[df == "-1"] = "|"
        print("".join(df[df.columns[0]].values.flatten().tolist())+"\t"+state.columns[0])
        
    def reward(self, arm, quiet=False, sample_id=None):
        import subprocess as sb
        if (str(sample_id) == "None"):
            from random import sample
            sample_id = sample(range(np.shape(self.P.values)[1]), 1)[0]
        initial = utils.binarize_via_histogram(self.P[[self.P.columns[sample_id]]])
        initial = initial.loc[list(filter(lambda x : x in initial.index, self.genes))].dropna()
        initial.columns = ["initial"]
        if (not quiet):
            self.print_state(initial)
        sig = self.binarized_X[[self.binarized_X.columns[arm]]]
        sig = sig.loc[list(filter(lambda x : x in sig.index, self.genes))].dropna()
        sig.columns = ["drug"]
        if (not quiet):
            self.print_state(sig)
        masked = utils.apply_mask(initial, sig).dropna()
        if (not quiet):
            self.print_state(masked)
        observations_name = "observations_prediction"
        observations_fname = self.path_to_grn+"/"+observations_name+".spec"
        length, solmax = 40, 1
        experiments = [{"cell": "Cell", "dose": "NA", "pert": "arm "+str(arm), "ptype": "trt_cp", "itime": "NA", "perturbed_oe": [], "perturbed_ko": [], "exprs": {"Initial": {"step": 0, "sig": masked}}}]
        N = utils.write_observations_file(observations_fname, length, self.perts, self.genes, experiments, verbose=False)
        assert N > 0
        ko_exists = any([v in self.perts for v in ["-", "+-", "-+"]])
        fe_exists = any([v in self.perts for v in ["+", "+-", "-+"]])
        cmd = "cd "+self.path_to_grn.split("examples/")[0]+"Python ; python solve.py launch "+self.path_to_grn.split("/")[-2]+" --model "+self.grn_name.split("/")[-1].split(".net")[0]
        cmd += " --experiments "+observations_name+" --q0 Initial1 --nstep "+str(length)+" --solmax "+str(solmax)
        cmd += (" --KO KnockDown1" if (ko_exists) else "")+(" --FE OverExpression1" if (fe_exists) else "")
        cmd += " --modelID 0 --steadyStates 1"
        output = sb.check_output(cmd, shell=True)
        try:
            treated, _ = utils.get_state(output, length, self.genes, solmax=solmax)
            if (not quiet):
                self.print_state(treated)
                #self.print_state(self.binarized_D)
            score = self.func(treated, self.binarized_D)
        except:
            print("/!\ score not computable")
            score = 0
        if (not quiet):
            print("Arm #"+str(arm)+" "+self.names[arm]+": score = "+str(score)+" (true: "+str(self.scores[arm])+")")
        return score

######################################

class simple_epilepsy(drug_repurposing):
    def __init__(self, X, scores, theta, problem_args):
        super(simple_epilepsy, self).__init__(X, scores, theta, problem_args)
        N, K = X.shape
        assert K in [10, 175]
        self.name = "simple_epilepsy_K="+str(K)
        score_file = "rewards_cosine_"+str(K)+"drugs_18samples.csv"
        self.rewards = pd.read_csv(self.dr_folder+score_file, header=0, index_col=0).values

    def reward(self, arm, quiet=False, sample_id=None):
        if (str(sample_id) == "None"):
            from random import sample
            sample_id = sample(range(self.rewards.shape[0]), 1)[0]
        score = float(self.rewards[sample_id, arm])
        if (not quiet):
            print("Arm #"+str(arm)+" "+self.names[arm]+": score = "+str(score)+" (true="+str(self.scores[arm])+")")
        return score
