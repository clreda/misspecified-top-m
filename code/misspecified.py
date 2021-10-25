# coding: utf-8

methods=["LUCB", "LinGapE", "LinGame", "MisLid", "TrueUniform"]
complexity_limit=1e8

import gc
import numpy as np
import utils
import random
import time

from learners import AdaHedge, Greedy

#########################################
## TOP-CLASS

class Misspecified(object):
    def __init__(self, method_args):
        for k in method_args:
            setattr(self, k, method_args[k])

    def clear(self):
        self.rewards = []
        self.samples = []
        self.t = 0
        self.na = []

    def sample(self, problem, candidates):
        if (not len(self.na)):
            self.na = [0]*(problem.X.shape[1])
        for arm in candidates:
            self.rewards.append(problem.reward(arm))
            self.samples.append(arm)
            self.t += 1
            self.na[arm] += 1

    def run(self, problem, nsimu, run_id=None):
        active_arms = np.array([0]*(problem.X.shape[1]))
        complexity = [None]*nsimu
        regret = [None]*nsimu
        linearity = [None]*nsimu
        running_time = [None]*nsimu
        for simu in range(nsimu):
            self.clear()
            starting_time = time.time()
            active_arms_, complexity_, regret_, linearity_ = self.apply(problem)
            running_time[simu] = time.time()-starting_time
            complexity[simu] = complexity_
            regret[simu] = regret_
            active_arms[active_arms_] += 1
            linearity[simu] = linearity_
            R, C, L = [round(np.mean(x[:simu+1]),2) for x in [regret, complexity, linearity]]
            id_ = simu + 1 if run_id is None else run_id
            print("It. #"+str(id_)+":\tR="+str(regret_)+" ("+str(R)+")\tC="+str(complexity_)+" ("+str(C)+")\tL="+str(linearity_)+" ("+str(L)+")")
            gc.collect()
        return active_arms/float(nsimu), complexity, regret, linearity, running_time

#########################################
## TRUE UNIFORM

class TrueUniform(Misspecified):
    def __init__(self, method_args):
        assert "n" in method_args
        self.name = "TrueUniform"
        method_args["beta"] = method_args["beta_linear"]
        super(TrueUniform, self).__init__(method_args)

    def update(self, problem, candidates, cumsum, means):
        candidates.reverse()
        for ia, a in enumerate(candidates):
            i_a = -(ia+1)
            cumsum[a] += self.rewards[i_a]
            means[a] = cumsum[a]/float(self.na[a])
        return cumsum, means

    def stopping_rule(self):
        if (self.t % 1000 == 0):
            print("t="+str(self.t)+" B(t)="+str(self.B)+" thres(t)="+str(self.epsilon)+" stop="+str(self.B < self.epsilon))
        return (self.B < self.epsilon) or (self.t > complexity_limit)

    def apply(self, problem):
        X = problem.X
        N, K = X.shape
        cumsum = [0]*K
        means = [0]*K
        self.T_init = 1
        for _ in range(self.T_init):
            for a in range(K):
                self.sample(problem, [a])
        cumsum, means = self.update(problem, [a for _ in range(self.T_init) for a in range(K)], cumsum, means)
        gap = lambda a, b, t: means[a]-means[b]+np.sqrt(2*self.beta(t, na))*(1/np.sqrt(self.na[a])+1/np.sqrt(self.na[b]))
        stop = False
        while(not stop):
            candidate = [self.t%K]
            self.sample(problem, candidate)
            cumsum, means = self.update(problem, candidate, cumsum, means)
            J = utils.argmax_m(means, self.m)
            b_t = J[-1]
            c_t = utils.randf([gap(c,b_t, self.t) for c in range(K) if (c not in J)], 1, np.max)[0]
            self.B = gap(c_t,b_t, self.t)
            stop = self.stopping_rule()
        J = utils.argmax_m(means, self.m)
        mu_m = utils.max_m(problem.scores, self.m)[-1]
        R = 1-int(all([problem.scores[a]>=mu_m-self.epsilon for a in J]))
        return J, self.t, R, 0.

#########################################
## LINGAME [Degenne et al., ICML'20]

class LinGame(Misspecified):
    def __init__(self, method_args):
        self.learner_type = AdaHedge #Only one implemented for LinGame 
        self.T_init = 1
        self.name = "LinGame"
        assert "beta_linear" in method_args
        method_args["beta"] = method_args["beta_linear"]
        self.m = method_args["m"]
        assert self.m == 1
        assert method_args["sigma"] == 1.
        self.lambda_ = method_args["sigma"]/float(20.)
        assert self.lambda_ > 0
        super(LinGame, self).__init__(method_args)

    def stopping_rule(self):
        if (self.t % 1000 == 0):
            print("t="+str(self.t)+" B(t)="+str(self.B)+" thres(t)="+str(self.epsilon)+" stop="+str(self.B > self.epsilon))
        return (self.B > self.epsilon) or (self.t > complexity_limit)

    def best_answer(self, means):
        i_star = utils.argmax_m(means, self.m)
        return i_star

    def update(self, problem, candidates, Vinv, b):
        c = 0.
        Vinv, b, means, theta, _ = utils.update_misspecified(problem, candidates, Vinv, b, c, self.na, self.rewards)
        return Vinv, b, means, theta

    def apply(self, problem):
        assert problem.name == "gaussian"
        self.B, self.epsilon = -float("inf"), 0.
        N, K = problem.X.shape
        means = [0]*K
        learners, sum_w = {}, np.zeros(K)
        Vinv, b = 1/(self.lambda_)**2*np.eye(N), np.zeros((K, 1))
        stop = False
        for _ in range(self.T_init):
            for a in range(K):
                self.sample(problem, [a])
        if (self.T_init > 0):
            Vinv, b, means, theta = self.update(problem, [a for _ in range(self.T_init) for a in range(K)], Vinv, b)
        while (not stop):
            i_star = self.best_answer(means)
            learner = learners.get(tuple(i_star), None)
            if (str(learner) == "None"):
                learner = self.learner_type(K)
            w_t = learner.act()
            sum_w += w_t
            eta = np.zeros((K,1))
            means_alt, closest, val, _ = utils.closest_alternative(problem, b, means, theta, eta, w_t, 0., i_star, constraint="L_inf")
            if ("AdaHedge" in str(self.learner_type)):
                mu = np.array(means).flatten()
                lambda_ = np.array(means_alt).flatten()
                delta = -utils.optimistic_gradient(problem, Vinv, mu, lambda_, self.na, self.t, self.M, 0., self.gain_type if (str(self.gain_type)!="None") else "linear")
            else:
                raise ValueError("Unknown learner type.")
            learner.incur(delta)
            di_learner = {}
            di_learner.setdefault(tuple(i_star), learner)
            learners.update(di_learner)
            candidate = utils.tracking_rule(w_t, sum_w, self.na, self.t, self.tracking_type if (str(self.tracking_type)!="None") else "D") #"D" tracking type is default
            self.sample(problem, candidate)
            Vinv, b, means, theta = self.update(problem, candidate, Vinv, b)
            means_alt, closest, val, _ = utils.closest_alternative(problem, b, means, theta, eta, self.na, 0., self.best_answer(means), constraint="L_inf")
            self.B = val
            self.epsilon = self.beta(self.t, self.na)
            stop = self.stopping_rule()
        J = self.best_answer(means)
        mu_m = utils.max_m(problem.scores, self.m)[-1]
        R = 1-int(all([problem.scores[a]>=mu_m-0. for a in J]))
        return J, self.t, R, 1. # data is assumed linear

#########################################
## Our paper 

class MisLid(Misspecified):
    def __init__(self, method_args):
        self.name = "MisLid"
        if (method_args["learner_name"] == "AdaHedge"):
            self.learner_type = AdaHedge
        elif (method_args["learner_name"] == "Greedy"):
            self.learner_type = Greedy
        else:
            raise ValueError("Learner '"+method_args["learner_name"]+"' not implemented.")
        self.T_init = 1 
        assert method_args["sigma"] == 1
        assert "beta_linear" in method_args
        method_args["beta"] = method_args["beta_linear"]
        self.m = method_args["m"]
        assert method_args["sigma"] == 1.
        self.lambda_ = method_args["sigma"]/float(20.)
        assert self.lambda_ > 0
        self.constraint = "L_inf"
        assert self.constraint in ["L_inf", "L1"]
        self.cnorm = lambda x : utils.cnorm(x, norm=self.constraint)
        super(MisLid, self).__init__(method_args)

    def stopping_rule(self, quiet=True):
        if (not quiet):
            print("t="+str(self.t)+" B(t)="+str(self.B)+" thres(t)="+str(self.epsilon)+" stop="+str(self.B > self.epsilon))
        return (self.B > self.epsilon) or (self.t > complexity_limit)

    def best_answer(self, means):
        J = utils.argmax_m(means, self.m)
        return J

    def update(self, problem, candidates, Vinv, b, Vinv_val=None):
        return utils.update_misspecified(problem, candidates, Vinv, b, self.c, self.na, self.rewards, Vinv_val=Vinv_val)

    def apply(self, problem, precision=1e-7):
        ## Only Gaussian problems with l_infty-norm
        assert problem.name == "gaussian"
        self.B, self.epsilon = -float("inf"), 0.
        N, K = problem.X.shape
        means = [0]*K
        b = np.zeros((K,1))
        learners, sum_w = {}, np.zeros(K)
        stop = False
        L = float(np.max([self.cnorm(problem.X[:,a]) for a in range(K)]))
        self.x = 2*L

        A = problem.X.T
        print("minimum eigenvalue of A^TA: {}".format(np.linalg.eigvals(A.T.dot(A)).min()))
        
        ## Consider a (C-approximate) barycentric spanner
        F = utils.barycentric_spanner(problem.X, C=1, quiet=True)
        ## Uncomment the next two lines to get regularized version of design matrix instead
        #F = range(K)
        #self.T_init = 0
        candidates, maxiter = [], len(F)*(self.T_init+1)
        from scipy.sparse.linalg import eigsh
        get_eigmin = lambda M : eigsh(M, k=1, which="SM")[0][0]
        V = np.zeros((N,N))
        alternative_arms = []  # arms in the recent closest alternatives
        stopping_rule_time = max(5*N, 10)  # next time at which the sampling rule is checked
        while (self.t < maxiter): # t0 initialization phases to check V_t0 >= x Id
            ## Round-Robin procedure
            for a in F:
                self.sample(problem, [a])
                V += problem.X[:,a].dot(problem.X[:,a].T)
            candidates += F
            detVxId = np.linalg.slogdet(V-self.x*np.eye(N))
            detVxId = detVxId[0]*detVxId[1]
            detV = np.linalg.slogdet(V)[0]*np.linalg.slogdet(V)[1]
            if (np.exp(detVxId) >= 0 and np.exp(detV) > 0):
                break
        if (self.T_init > 0):
            Vinv = np.linalg.pinv(V)
            Vinv, b, means, theta, eta = self.update(problem, candidates, Vinv, b, Vinv_val=Vinv)
        else:
            ## Regularized version (necessary for invertible design matrices)
            ## In practice, never used
            Vinv = 1/float(self.lambda_**2)*np.eye(N)
            Vinv, b, means, theta, eta = self.update(problem, candidates, Vinv, b)
            V = np.linalg.pinv(Vinv)
            self.x = get_eigmin(V)
        while (not stop):
            S_t = self.best_answer(means)  # current best answer set
            
            if (self.multi_learners):  # Several learners
                learner = learners.get(tuple(S_t), None)  # get step from learner associated with best answer
            else:  # Only one learner
                learner = learners.get(0, None)  # get step from learner associated with best answer
            if (str(learner) == "None"):
                learner = self.learner_type(K)
            
            if (self.learner_name == "AdaHedge"):
                # query the learner
                w_t = learner.act() 
                # get closest alternative to current means - Best-Response
                means_alt, closest, val, alternative_arms = utils.closest_alternative(
                    problem, b, means, theta, eta, w_t, self.c,  S_t,
                    constraint=self.constraint, subsample=self.subsample, alternative_arms=alternative_arms)
            elif (self.learner_name == "Greedy"):
                # get closest alternative to current means - FTL
                means_alt, closest, val, alternative_arms = utils.closest_alternative(
                    problem, b, means, theta, eta, self.na, self.c,  S_t,
                    constraint=self.constraint, subsample=self.subsample, alternative_arms=alternative_arms)
            else:
                raise ValueError("Unknown learner type.")

            # optimism
            mu = np.array(means).flatten()
            lambda_ = np.array(means_alt).flatten()
            delta = -utils.optimistic_gradient(problem, Vinv, mu, lambda_, self.na, self.t, self.M, self.c,
                self.gain_type if (str(self.gain_type)!="None") else "misspecified")
            # update learner
            learner.incur(delta)  
            di_learner = {}
            di_learner.setdefault(tuple(S_t) if (self.multi_learners) else 0, learner)

            if (self.learner_name == "Greedy"):
                # pull argmin of the loss
                w_t = learner.act()

            nb_samples = 1
            if self.subsample:
                nb_samples = N
            for sample_id in range(nb_samples):  # Sample the same arm several times
                sum_w += w_t
                learners.update(di_learner)
                candidate = utils.tracking_rule(w_t, sum_w, self.na, self.t, self.tracking_type if (str(self.tracking_type)!="None") else "S") # default is "S"
                self.sample(problem, candidate)
                # update sum of rewards, estimated parameters and means
                Vinv, b, means, theta, eta = self.update(problem, candidate, Vinv, b)
            # Stopping rule test
            if (((self.t > stopping_rule_time) and self.subsample) or (not self.subsample)):
                means_alt, closest, val, alternative_arms = utils.closest_alternative(problem, b, means, theta, eta, self.na,
                    self.c, self.best_answer(means), constraint=self.constraint, subsample=False, alternative_arms=alternative_arms)
                self.B = val
                self.epsilon = self.beta(self.t, self.na, x=self.x)
                stop = self.stopping_rule(quiet=(self.t % 1000 != 0))  # check stopping rule
                #stop = self.stopping_rule(quiet=False)
                stopping_rule_time = max(self.t, stopping_rule_time)*self.geometric_factor
        J = self.best_answer(means)
        mu_m = utils.max_m(problem.scores, self.m)[-1]
        R = 1-int(all([problem.scores[a] >= mu_m-0. for a in J]))
        return J, self.t, R, 0.  # This algorithm assumes data is not linear and is not c-agnostic

#########################################
## LUCB [Kalyanakrishnan et al., ICML'12]

class LUCB(Misspecified):
    def __init__(self, method_args):
        self.T_init = 1
        self.name = "LUCB"
        assert "beta_classical" in method_args
        method_args["beta"] = method_args["beta_classical"]
        super(LUCB, self).__init__(method_args)

    def greedy(self, problem, b_t, c_t):
        variances = [self.na[a] for a in [b_t,c_t]]
        return [[b_t,c_t][utils.randf(variances, 1, np.min)[0]]]

    def update(self, problem, candidates, cumsum, means):
        candidates.reverse()
        for ia, a in enumerate(candidates):
            i_a = -(ia+1)
            cumsum[a] += self.rewards[i_a]
            means[a] = cumsum[a]/float(self.na[a])
        return cumsum, means

    def stopping_rule(self):
        if (self.t % 2000 == 0):
            print("t="+str(self.t)+" B(t) = "+str(self.B))
        return (self.B <= self.epsilon) or (self.t > complexity_limit)

    def apply(self, problem):
        self.B = float("inf")
        N, K = problem.X.shape
        self.I = [None]*K
        cumsum = [0]*K
        means = [0]*K
        w = lambda t, a : (0.5*self.beta(t, self.na)**2)/np.sqrt(self.na[a])
        stop = False
        for _ in range(self.T_init):
            for a in range(K):
                self.sample(problem, [a])
        if (self.T_init > 0):
            cumsum, means = self.update(problem, [a for _ in range(self.T_init) for a in range(K)], cumsum, means)
        while (not stop):
            J = utils.argmax_m(means, self.m)
            notJ = [a for a in range(K) if (a not in J)]
            lbounds = [means[a]-w(self.t, a) for a in J]
            min_idx = utils.randf(lbounds, 1, np.min)[0]
            b_t = J[min_idx]
            bt_indices = [means[c]-means[b_t]+w(self.t,c)+w(self.t,b_t) for c in notJ]
            max_idx_idx = utils.randf(bt_indices, 1, np.max)[0]
            c_t = notJ[max_idx_idx]
            candidate = self.greedy(problem, b_t, c_t)
            self.sample(problem, candidate)
            cumsum, means = self.update(problem, candidate, cumsum, means)
            self.B = bt_indices[max_idx_idx]
            self.I = [[means[a]-w(self.t, a), means[a]+w(self.t, a)] for a in range(K)]
            stop = self.stopping_rule()
        J = utils.argmax_m(means, self.m)
        mu_m = utils.max_m(problem.scores, self.m)[-1]
        R = 1-int(all([problem.scores[a]>=mu_m-self.epsilon for a in J]))
        return J, self.t, R, 0.

#########################################
## LINGAPE [Xu et al., AISTATS'18]

class LinGapE(Misspecified):
    def __init__(self, method_args):
        self.T_init = 1
        assert "beta_linear" in method_args
        self.name = "LinGapE"
        method_args["beta"] = method_args["beta_linear"]
        super(LinGapE, self).__init__(method_args)

    def stopping_rule(self):
        if (self.t % 2000 == 0):
            print("t="+str(self.t)+" B(t) = "+str(self.B))
        return (self.B <= self.epsilon) or (self.t > complexity_limit)

    def greedy(self, problem, b_t, c_t, Vinv):
        K = problem.X.shape[1]
        direction = problem.X[:, b_t]-problem.X[:, c_t]
        uncertainty = [float(utils.mahalanobis(direction, utils.sherman_morrison(Vinv, problem.X[:,i]))) for i in range(K)]
        a = utils.randf(uncertainty, 1, np.min)[0]
        return [a]

    def optimized(self, problem, b_t, c_t, Vinv):
        p = self.ratio.get((b_t,c_t))
        if (not len(p)):
            ## https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with-linear-equality-constraints-basis-pu
            from scipy.optimize import linprog
            X = problem.X
            Aeq = np.concatenate((X, -X), axis=1)
            beq = (X[:,b_t]-X[:,c_t])
            F = np.ones((2*K, ))
            bounds = [(0, float("inf"))]*(2*K)
            solve = linprog(F, A_eq=Aeq, b_eq=beq, bounds=bounds)
            x = solve.x
            w = x[:K]-x[K:]
            assert solve.status == 0
            p = np.abs(w)
            p /= np.linalg.norm(w, 1)
            self.ratio.setdefault((b_t,c_t), p)
        samplable_arms = [i for i in range(problem.X.shape[1]) if (float(p[i]) > 0)]
        a = samplable_arms[self.randf([float(self.na[i]/float(p[i])) for i in samplable_arms], 1, np.min)]
        return [a]

    def update(self, problem, candidates, Vinv, b):
        candidates.reverse()
        for ia, a in enumerate(candidates):
            i_a = -(ia+1)
            Vinv = utils.sherman_morrison(Vinv, problem.X[:, a])
            b += self.rewards[i_a]*problem.X[:, a]
        theta = Vinv.dot(b)
        means = np.array(theta.T.dot(problem.X)).flatten().tolist()
        return Vinv, b, means

    def apply(self, problem, greedy_sampling=True, lambda_val=1.):
        self.B = float("inf")
        self.ratio = {}
        N, K = problem.X.shape
        Vinv = 1/float(lambda_val**2)*np.matrix(np.eye(N)).reshape((N,N))
        b = np.matrix(np.zeros(N)).reshape((N,1)) 
        means = [0]*K
        self.Cbeta = lambda t, n : np.sqrt(2*self.beta(t,n))
        w = lambda t, c, a, Vinv : float(self.Cbeta(t, self.na[c]+self.na[a])*utils.mahalanobis(problem.X[:,c]-problem.X[:,a], Vinv))
        w_ind = lambda t, a, Vinv : float(self.Cbeta(t, self.na[a])*utils.mahalanobis(problem.X[:,a], Vinv))
        Bidx = lambda t, i, j, Vinv : float(means[i]-means[j]+w(t, i, j, Vinv))
        stop = False
        for _ in range(self.T_init):
            for a in range(K):
                self.sample(problem, [a])
        if (self.T_init > 0):
            Vinv, b, means = self.update(problem, [a for _ in range(self.T_init) for a in range(K)], Vinv, b)
        while (not stop):
            J = utils.argmax_m(means, self.m)
            notJ = [a for a in range(K) if (a not in J)]
            indices = [[means[c]-means[a]+w(self.t, c, a, Vinv) for c in notJ] for a in J]
            max_idx = utils.randf([np.max(ids) for ids in indices], 1, np.max)[0]
            b_t = J[max_idx]
            max_idx_idx = utils.randf(indices[max_idx], 1, np.max)[0]
            c_t = notJ[max_idx_idx]
            if (greedy_sampling):
                candidate = self.greedy(problem, b_t, c_t, Vinv)
            else:
                candidate = self.optimized(problem, b_t, c_t, Vinv)
            self.sample(problem, candidate)
            Vinv, b, means = self.update(problem, candidate, Vinv, b)
            self.B = float(indices[max_idx][max_idx_idx])
            stop = self.stopping_rule()
        J = utils.argmax_m(means, self.m)
        mu_m = utils.max_m(problem.scores, self.m)[-1]
        R = 1-int(all([problem.scores[a]>=mu_m-self.epsilon for a in J]))
        return J, self.t, R, 1.
