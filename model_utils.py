import os
from typing import Tuple, List, Iterable, Callable

import math
import time
import numpy as np
import pandas as pd
from scipy.stats import halfcauchy, invgamma
import cvxpy as cp
from matplotlib import pyplot as plt
from scipy import stats

from scipy.stats import multivariate_normal
from numpy.random import default_rng
from itertools import chain, combinations, product

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error


# For Testing only
from sklearn.datasets import make_spd_matrix
import numpy as np
import cvxpy as cvx
from itertools import combinations


# - - - - - - - - - - - - - - - - - - - - -
#
#       HELPER FUNCTIONS
#
# - - - - - - - - - - - - - - - - - - - - - 
def powerset(x:Iterable):
    '''
    Powerset (set of subsets) for a given iterable x incl. ∅
    '''
    s = list(x)
    powerSet = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    return list(powerSet)


def d_hamming(x:np.array, z:np.array):
    '''
    Returns the Hamming distance between to vectors `x` and `z`.
    '''
    x = np.array(x)
    z = np.array(z)
    
    assert x.shape==z.shape, "Input arrays `x` and `z` must have the same shape"
    
    return sum(np.abs(x - z))


def f_bin(x:np.array, X:np.array, y:np.array, maxHamm:int=2, metric:Callable=d_hamming, varInfFac:float=0.1, seed:int=0) -> float:
    '''
    Emulates the function mapping {0,1}^d -> R
    - x: numpy argument (suggestion)
    - X: design matrix with observed binary vectors
    - y: observed target values for corresponding design matrix
    - maxHamm: maximum Hamming distance up to which function values f(x_tilde) of approximate arguments are to be considered
    - varInfFac: control by how much standard deviation is increased in the hamming distance
    '''
    
    assert set(np.unique(X))=={0,1}, '(Complete) design matrix `X` is expected to be a binary matrix.'
    assert len(X)==len(y), f"Lengths of input data `X` and `y` must coincide but len(X)={len(X)}!={len(y)}=len(y)."
    
    d_vec = np.apply_along_axis(d_hamming, arr=X, z=x, axis=1)
    min_d = min(d_vec)

    # TEST
    #print('min_d:', min_d)
    #print('y_std: ', np.std(y[d_vec==min_d]))
    
    if(min_d in range(maxHamm+1)):
        mu_, sigma = y[d_vec==min_d].mean(), np.std(y[d_vec==min(d_vec)])
        y_rnd = np.random.normal(loc=mu_, scale=sigma*(0.2 + varInfFac*min_d), size=1)
    else:
        y_rnd = np.random.normal(loc=np.mean(y), scale=np.std(y), size=1)
      
    # TEST
    #print('y_act: ', sigma*(0.1 + varInfFac*min_d))
        
    return y_rnd[0]




# - - - - - - - - - - - - - - - - - - - - -
#
#       ORACLE
#
# - - - - - - - - - - - - - - - - - - - - - 
class MaxQueriesExceeded(Exception):
    def __init__(self, message):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        
        

class Oracle:
    def __init__(self, fun, order:int=2, sigma_2:float=0.0, N_total:int=1000, seed:int=0):
        '''
        Noise terms are sampled at initialization to s.t. sampled function values f(x_i) are 
        reporducible regardless how the oracle is sampled.
        '''
        assert isinstance(fun, Callable), "Input `f` must be a callable function."
        assert isinstance(order, int) and 1<=order<6, "Maximum `order` of interaction terms should be 5."
        assert isinstance(sigma_2, float) and sigma_2>=0, "Noise variance parameter `sigma_2` must be a non-negative float."
        assert isinstance(seed, int), "Seed for random draws `seed` must be an integer."
        assert isinstance(N_total, int), "Maximum number of queries `N_total` must be an positive integer."
        
        self.fun = fun
        self.order = order
        self.sigma_2 = sigma_2
        self.seed = seed
        self.N_total = N_total
        self.N_current = 0
        
        if(self.sigma_2 > 0):
            np.random.seed(self.seed)
            self.eps = np.random.normal(loc=0, scale=np.sqrt(self.sigma_2), size=self.N_total)
        else:
            self.eps = np.zeros(self.N_total)
    
    def __expandX__(self, x:np.array) -> np.array:
        '''
        Expand a binary input vector `x` from the original input format with indices {1,...,d} to
        {0} (intercept), {1,...,d} (1st-order coefs), {... (d over 2) ... } (2nd-order effects)
        '''
        
        # 
        matrixInput = len(x.shape)==2 and len(x)>1
        
        # transform
        if(matrixInput):
            highOrdMats = [np.ones(len(x)).reshape(-1,1), x]
            # o-th order : CORRECT
            for o in range(2, self.order+1):
                highOrdMats.append(np.stack([np.prod([x[:,pair[i]] for i in range(o)], axis=0) for pair in powerset(range(x.shape[1])) if len(pair)==o], axis=1))

            #print('highOrdMats : ', highOrdMats)
            x = np.concatenate(highOrdMats, axis=1)
        else:
            # o-th order : CORRECT
            x = np.array(x, dtype=np.float64)
            highOrdMats = [x]
            for o in range(2, self.order+1):
                highOrdMats.append(np.array([np.prod([x[pair[i]] for i in range(o)]) for pair in powerset(range(len(x))) if len(pair)==o]))

            #print(highOrdMats)
            x = np.concatenate(([np.array([1])] + highOrdMats), axis=0)
        
        return x
    
    def f(self, x:np.array):
        '''
        Returns (noisy) function value f(x). 
        If `noiseFlag` is set to True, f(x) + eps with eps ~ N(f(x), sigma_2) is returned.
        '''
    
        # expand raw input
        #x = self.__expandX__(x)
        
        matInputFlag = len(x.shape)==2 and len(x)>1
        
        # check if within remaining query budget
        n_req = len(x) if matInputFlag else 1
        
        if(self.N_current + n_req > self.N_total):
            raise MaxQueriesExceeded(f"Maximum number of queries `N_total`={self.N_total} would be exceeded with these `n_req`={n_req} additional requests given that `N_current=`{self.N_current}.")
        
        # compute (noisy) function value
        if(matInputFlag):
            fx =  self.fun(x) + self.eps[self.N_current:self.N_current+len(x)] if (self.sigma_2 > 0) else np.zeros(len(x))
        else:
            fx =  self.fun(x) + (float(self.eps[self.N_current]) if (self.sigma_2 > 0) else 0)
        
        # update budget
        self.N_current += n_req
        
        return fx
    

# - - - - - - - - - - - - - - - - - - - - -
#
#       SPARSE BAYESIAN REGRESSION
#
# - - - - - - - - - - - - - - - - - - - - - 

class SparseBayesReg:
    def __init__(self, N_total:int, order:int=2, seed:int=0, burnin:int=0, thinning:int=1, d_MAX:int=20):

        assert isinstance(seed, int), "`seed` must be an integer."
        assert isinstance(burnin, int) and burnin>=0, "`burnin` must be a non-negative integer."
        assert isinstance(thinning, int) and thinning>=1, "`thinning` must be an positive integer."
        
        # - assignment
        self.N_total = N_total
        self.order = order
        self.seed = seed
        self.burnin = burnin
        self.thinning = thinning
        self.d_MAX = d_MAX
        
    def setXy(self, X:np.array, y:np.array) -> None:
        '''
        Setup of design matrix X (standardized, incl. leading 1-column and quadratic terms) and target vector y (transalted to E[y]=0)
        '''
        assert sum(X[:,0])!=len(X), "Provide the design matrix X without adding a leading 1-column (for intercept)"
        
        self.d = X.shape[1]
        self.p = sum([math.comb(self.d, k) for k in range(0,self.order+1)])
        
        assert isinstance(self.d, int) and 0<self.d<=self.d_MAX, f"The inferred dimension `d`={self.d} should be smaller than {self.d_MAX}"
        
        X = self.__expandX__(X)         # arbitrary interaction effects of order k
        X = self.__standardizeX__(X)
        X = self.__interceptColumnX__(X)
        self.X = X
        
        #y = self.__translateY__(y)  # TEST
        self.y = y
        
    def setX(self, X:np.array) -> None:
        '''
        Setup of design matrix X(_new) for prediction only.
        '''
        #assert sum(X[:,0])!=len(X), "Provide the design matrix X without adding a leading 1-column (for intercept)"
        assert X.shape[1]==self.d, f"Input matrix `X` has {X.shape[1]} columns but {self.d} was expected. Provide input with original dimension `d`."

        X = self.__expandX__(X)
        X = self.__standardizeX__(X, trainMode=False)
        X = self.__interceptColumnX__(X)
        
        return X
    
    def __expandX__(self, X:np.array) -> None:
        '''
        Given a (binary) design matrix X, it appends pairwise products of columns and appends to the design matrix X;
        *excluding* the leading intercept column of 1's
        '''
        
        assert self.order<=X.shape[1], f"`Order` of interaction terms can be at most number of columns of design matrix `X`. Lower the `order` to `X.shape[1]`={X.shape[1]}."
        if(self.order==X.shape[1]):
            raise np.linalg.LinAlgError("Setting `order` equal to the number of columns of the design matrix will cause parameters to be unidentifiable.")
        
        # append to arbitrary terms
        xList = [X]
        # o-th order : CORRECT
        for k in range(2, self.order+1):
            xList.append(np.stack([np.prod([X[:,pair[i]] for i in range(k)], axis=0) for pair in powerset(range(X.shape[1])) if len(pair)==k], axis=1))

        #print('highOrdMats : ', highOrdMats)
        X = np.concatenate(xList, axis=1)
        
        return X

    def __standardizeX__(self, X:np.array, trainMode:bool=True, EPS_NUM:float=0.1) -> None:
        '''
        Standardizes (translates & rescales) the columns of the design matrix (input matrix) 
        - EPS_NUM: lower bound for column-wise standard deviation term; ensures numerical stability for standardizations (rescaling)
        '''

        
        assert X.shape[1]==self.p-1, "The given design matrix includes a leading 1-column; unclear if it is a legitimiate (coincidental) feature or leading 1 column was already incldued"

        if(trainMode):
            X_mu, X_sigma = X.mean(axis=0), X.std(axis=0).clip(EPS_NUM)
            self.X_mu = X_mu
            self.X_sigma = X_sigma  #np.sqrt(len(X)) * X_sigma  # corrected for sqrt(n)
        else:
            assert (self.X_mu is not None) and (self.X_sigma is not None), "`X_mu` and `X_sigma` must be pre-computed."
        
        X = (X - self.X_mu) / self.X_sigma
        
        return X
        
    def __interceptColumnX__(self, X:np.array) -> np.array:
        '''
        Adds a leading vector of 1s to the binary matrix X (of 1st and 2nd order interactions)
        '''
        
        X = np.concatenate((np.ones_like(X[:,0]).reshape(-1,1), X), axis=1)

        assert X.shape[1]==self.p, "Inconsistent number of columns after adding leading 1-col"
        
        return X

    def __translateY__(self, y:np.array, trainMode:bool=True) -> None:
        '''
        LEGACY
        Translation of the target vector y such that priori condition E[y]=0 is satisfied.
        (No rescaling to unit variance is applied, though.)
        '''
        X = self.X

        assert len(X) == len(y), "Length of target vector y does not coincide with design matrix X"

        # Standardize y's
        if(trainMode):
            self.y_mu = np.mean(y)
        else:
            assert self.y_mu, "`y_mu` must be computed."
        
        # translation
        y = y - self.y_mu
        
        return y
        
    def add(self, X:np.array, y:float, fitFlag:bool=True) -> None:
        '''
        Appends new datapoint to X,y
        '''
        
        # YYY
        assert isinstance(X, np.ndarray), "New data `X` must be provided as a numpy array."
        
        # convert obs. vector (n=1)
        if len(X.shape)==1:
            X = X.reshape(1, len(X))
            if(isinstance(y, float) or isinstance(y, int)):
                y = np.array([y])
        
        assert X.shape[1]==self.d, "New data input should have column dim. `d`."
        assert len(X)==len(y), "Lengths of new data `X` and `y` do not coincide."
        
        # New
        #X = self.__expandX__(X)
        #X = self.__standardizeX__(X, trainMode=False)
        #X = self.__interceptColumnX__(X)
        
        # Newer
        X = self.setX(X)
        
        assert self.X.shape[1]==X.shape[1], "Number of columns of new data must coincide with design matrix `X`."
        
        # append to data
        self.X = np.concatenate((np.array(self.X), X), axis=0)
        self.y = np.concatenate((np.array(self.y), y), axis=0)
        
        # re-fit
        if(fitFlag):
            self.__fit__()
            
        return None
    
    def __mvg__(self, Phi, alpha, D):
        '''
        Sample multivariate Gaussian (independent of d) from NumPy
        Not used Rue (2001) or et. al. (2015) approaches on fast sampling mvg
        N(mean = S@Phi.T@y, cov = inv(Phi'Phi + inv(D))
        '''
        #assert len(Phi.shape)==2 and Phi.shape[0]==Phi.shape[1], "`Phi` must be a quadratic matrix."
        assert len(D.shape)==2 and D.shape[0]==D.shape[1], "`D` must be a quadratic matrix."
        
        S = np.linalg.inv(Phi.T @ Phi + np.linalg.inv(D))
        x = np.random.multivariate_normal(mean=((S @ Phi.T) @ y), cov=S, size=1)
        
        return x
    
    def sampleStandardizedAlpha(self, ) -> np.array:
        '''
        Samples posterior  ~ P( |X_tilde,y) from (most current) posterior distribution 
        parametrized by the standardized design matrix X_tilde.
        '''
        
        assert (self.alpha_mu is not None) and (self.alpha_cov is not None), "Posterior mean and covariance not available yet as the model been computed yet. Run `.fit(X,y)`."
        
        # sample
        alpha_post = np.random.multivariate_normal(mean = self.alpha_mu,
                                                   cov  = self.alpha_cov,
                                                   size = 1).reshape(-1)
        
        return alpha_post

    def sampleAlpha(self, ) -> np.array:
        '''
        Samples posterior  ~ P( |X,y) from (most current) posterior distribution 
        parametrized by the design matrix X.
        '''
        
        alpha_post = np.array(self.sampleStandardizedAlpha())
        
        # correct w.r.t. X_tilde = (X - X_mu) / X_sigma
        alpha_post[1:] /= self.X_sigma
        alpha_post[0]  -= self.X_mu @ alpha_post[1:] 
        
        return alpha_post
    
    def getMeanStandardizedAlpha(self, ) -> np.array:
        '''
        Posterior mean of alpha based on the row-wise standardized design matrix X_tilde
        '''
        
        return np.array(self.alpha_mu)
    
    def getMeanAlpha(self, ) -> np.array:
        '''
        Posterior mean of alpha based on the (actual) design matrix X.
        '''
        alpha_mu = np.array(self.getMeanStandardizedAlpha())
        
        # correct w.r.t. X_tilde = (X - X_mu) / X_sigma
        alpha_mu[1:] /= self.X_sigma
        alpha_mu[0]  -= self.X_mu @ alpha_mu[1:]  
        
        return alpha_mu
    
    # sbr.X_mu @ sbr.getMeanAlpha()[1:]
    
    def __fit__(self) -> None:
        '''
        Core of fitting procedure (on self.X, self.y)
        '''
        
        # D0
        self.n = len(self.X)
        
        # setup values
        alphas_out = np.zeros((self.p, 1))
        s2_out     = np.zeros((1, 1))
        t2_out     = np.zeros((1, 1))
        l2_out     = np.zeros((self.p, 1))

        # sample priors
        betas   = halfcauchy.rvs(size=self.p) 
        tau_2   = halfcauchy.rvs(size=1)                            
        nu      = np.ones(self.p) # ?
 
        sigma_2, xi = 1.0, 1.0
        
        # Gibbs sampler
        for k in range(self.N_total):
            sigma = np.sqrt(sigma_2)

            # alphas
            # - Sigma_star
            Sigma_star = tau_2 * np.diag(betas**2) # Sigma_star
            Sigma_star_inv = np.linalg.inv(Sigma_star)
            
            # - A
            A     = (self.X.T @ self.X) + Sigma_star_inv
            # - - add regularity to A if required
            if(np.linalg.cond(A) > 10_000):
                A += np.eye(A.shape[0])
            # - invert
            A_inv = np.linalg.inv(A)
            
            # - update posterior mean, cov
            self.alpha_mu  = A_inv @ self.X.T @ self.y
            self.alpha_cov = sigma_2 * A_inv
            
            # - alpha
            alphas = self.sampleStandardizedAlpha()
            
            # - sigma_2
            sigma_2 = invgamma.rvs(0.5*(self.n+self.p), scale=0.5*(np.linalg.norm((self.y - self.X @ alphas), 2)**2 + (alphas.T @ Sigma_star_inv @ alphas)))

            # - betas
            betas_2 = invgamma.rvs(np.ones(self.p), scale=(1. / nu) + (alphas**2)/(2. * tau_2 * sigma_2))
            betas = np.sqrt(betas_2)

            # - tau_2
            tau_2 = invgamma.rvs(0.5*(self.p+1), scale=1.0 / xi + (1. / (2. * sigma_2)) * sum(alphas**2 / betas**2), size=1)

            # - nu
            nu = invgamma.rvs(np.ones(self.p), scale=1.0 + 1. / betas_2, size=self.p)

            # - xi
            xi = invgamma.rvs(1.0, scale=1.0 + 1. / tau_2, size=1)
            
            # store samples
            if k > self.burnin:
                # - append
                if(k%self.thinning==0):
                    alphas_out = np.append(arr=alphas_out, values=alphas.reshape(-1,1), axis=1)
                    s2_out = np.append(s2_out, sigma_2)
                    t2_out = np.append(t2_out, tau_2)
                    l2_out = np.append(arr=l2_out, values=betas.reshape(-1,1), axis=1)

        # Clip 1st value
        self.alphas = alphas_out[:,1:]
        self.s2 = s2_out[1:]
        self.t2 = t2_out[1:]
        self.l2 = l2_out[1:]
        
    def getStandardizedAlphas(self) -> np.array:
        '''
        Returns current array of alpha posterior samples (based on the standardized design matrix X_tilde)
        '''
        return self.alphas
    
    def fit(self, X:np.array, y:np.array) -> None:
        '''
        Fitting the (initial) model on the data D0={X0,y0}
        '''
        assert len(X.shape)==2 and len(y.shape)==1, "Design matrix X and target vector y."
        assert X.shape[0]==len(y), f"Dimension of design matrix X and target vector y do not coincide: X.shape[0]={X.shape[0]}!={len(y)}=len(y)"
        assert len(X) < self.N_total, f"Implied `N_init`=len(X)={len(X)} exceeds `N_total`={self.N_total}."
        
        # setup
        self.setXy(X, y)
        
        # fitting
        self.__fit__()
        
    def predict(self, X:np.array, mode:str='mean') -> np.array:
        '''
        Obtain prediction
        '''
        assert mode in ['mean', 'post'], "`predict`ion from Bayesian sparse regression either by `mean` (MLE estimator of alpha) or randomly sampled from `post`erior."
        assert X.shape[1] == self.d, f"Format of input matrix wrong. Does not have `d`={self.d} columns but `X.shape[1]`={X.shape[1]}"
        
        # TraFo data
        X = self.setX(X)
        
        # apply model
        alpha_hat = self.getMeanStandardizedAlpha() if mode=='mean' else self.sampleStandardizedAlpha()
        
        # dot prod
        y_hat = X @ alpha_hat
        
        # revsert-translation
        #y_hat = y_hat + self.y_mu # TEST
        
        
        return y_hat
    

# - - - - - - - - - - - - - - - - - - - - -
#
#       Semi-Definite Programming (SDP
#
# - - - - - - - - - - - - - - - - - - - - - 

class SDP:
    def __init__(self, alpha:np.array, lambd:float=0.1, pen_ord:int=2, mode:str='min', d_MAX:int=20) -> List[np.array]:
        
        assert isinstance(mode, str), "Input `mode` must be a either `min` or `max`."
        assert mode in ['min', 'max'], f"Input `mode` is str. In addition, it must be a str either `min` or `max` but `{mode}` was provided."
        assert isinstance(lambd, float) and lambd >=0, "lambda (regularization parameter) must be non-negative scalar."
        assert pen_ord in [1, 2], "Penalty norm order `pen_ord` must be either 1 or 2."
        
        self.mode = mode
        self.lambd = lambd
        self.pen_ord = pen_ord
        self.alpha = alpha
        self.d_MAX = d_MAX
        
        # infer d(imension): possible since only quadratic terms possible
        self.p = len(self.alpha)
        dDict = {1+dLoc+math.comb(dLoc,2) : dLoc for dLoc in range(1,self.d_MAX+1)}
        if(self.p in dDict.keys()):
            self.d = dDict[self.p]
        else:
            assert False, f'Length of `alpha` is not a 1+d+binom(d,2) for any 1,2,...,{self.d_MAX}'
        assert isinstance(self.d, int), "Dimension `d` must be non-negative integer."
        
        # extract 1st/2nd order terms
        b = self.alpha[1:1 + self.d]  # 1st
        a = self.alpha[1 + self.d:]   # 2nd

        # get indices for quadratic terms
        idx_prod = np.array(list(combinations(np.arange(self.d), 2)))
        d_idx = idx_prod.shape[0]

        # check number of coefficients
        if len(a)!=d_idx:
            assert False, 'Number of Coefficients does not match indices!'

        # xAx-term
        A = np.zeros((self.d, self.d))
        for i in range(d_idx):
            A[idx_prod[i,0], idx_prod[i,1]] = 0.5 * a[i]
        A += A.T

        # bx-term
        bt = 0.5 * (b + A @ np.ones(self.d)).reshape((-1, 1))
        bt = bt.reshape((self.d, 1))
        At = np.vstack((np.append(0.25*A, 0.25*bt, axis=1), np.append(bt.T, 2.)))
        
        self.A  = A
        self.b  = b
        self.At = At
        self.bt = bt
        
    def run(self) -> np.array:
        '''
        Runs the BQP-relaxation, SDP-optimization, and extracts candidate x via geometric rounding.
        '''
        self.solve()
        self.decompose()
        return self.geometricRounding()
        
        
    def solve(self, ) -> None:
        '''Actual solver of the semi-definite programming solver.'''
        
        # SDP relaxation
        Xvar = cp.Variable((self.d+1, self.d+1), PSD=True)
        
        # - objective function
        if(self.mode=='min'):
            f0 = cp.Minimize(cp.trace((self.At @ Xvar)))
        else:
            f0 = cp.Maximize(cp.trace((self.At @ Xvar)))
        
        # - constraints
        constraints = [cp.diag(Xvar) == np.ones(self.d+1)]
        prob = cp.Problem(f0, constraints)
        prob.solve()
        
        self.Xvar = Xvar
        
    def decompose(self) -> None:
        '''
        Wrapper for stable Cholesky decomposition
        '''
        self.L = self.__stableCholesky__(eTol=1E-12)
    
    def __stableCholesky__(self, eTol:float=1E-10) -> np.array:
        '''
        Performs numerically stable Cholesky decomposition (by adding regularity to the matrix until PSD). 
        '''
        try:
            return np.linalg.cholesky(self.Xvar.value + eTol*np.eye(self.Xvar.value.shape[0]))
        except Exception as e:
            if(isinstance(e, np.linalg.LinAlgError)):
                return self.__stableCholesky__(10*eTol)
            else:
                pass
    
    def geometricRounding(self, k_rounds:int=100) -> np.array:
        '''
        Random geometric round and conversion to original space
        - k_rounds: number of iterations
        '''
        x_cand  = np.zeros((self.d, k_rounds))
        f_star  = np.zeros(k_rounds)

        for j in range(k_rounds):
            # rnd cutting plane vector (U on Sn) 
            r = np.random.randn(self.d+1)
            r /= np.linalg.norm(r, ord=2)
            
            # rnd hyperplane
            y_star = np.sign(self.L.T @ r)

            # convert solution to original domain and assign to output vector
            x_cand[:,j] = 0.5 * (1.0 + y_star[:self.d])
            f_star[j] = (x_cand[:,j].T @ self.A @ x_cand[:,j]) + (self.b @  x_cand[:,j])

            # Find optimal rounded solution
            if(self.mode=='min'):
                f_argopt = np.argmin(f_star)
            else:
                f_argopt = np.argmax(f_star)
            x_0      = x_cand[:,f_argopt]
            f_0      = f_star[f_argopt]

        return (x_0, f_0)
    
    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - 
#
#    DISCRETE BAYESIAN OPTIMIZATION HEURISTICS
#               (WRAPPER CLASSES)
#
# - - - - - - - - - - - - - - - - - - - - - - - - - 

class BOCS:
    variantList = ['SDP', 'SA']
    optModes = ['min', 'max']
    
    def __init__(self, variant:str='SDP', oracle:Oracle=None, N:int=1, B:int=0, T:int=1, seed:int=0,
                 lambd:float=0.5, pen_ord:int=2, mode:str='min'):
        
        assert variant in self.variantList, f"AFO `variant` must be in: {', '.join(variantList)}"
        
        self.variant = variant
        self.N = N
        self.seed = seed
        self.B = B
        self.T = T
        self.lambd = lambd
        self.pen_ord = pen_ord
        self.mode = mode
        self.oracle = oracle
        
        # init Sparse Bayesian Regression
        self.BayReg = SparseBayesReg(N_total=self.N, burnin=self.B, thinning=self.T, seed=self.seed)
        
    def fit(self, X:np.array, y:np.array) -> None:
        '''
        Delegate fit to bayesian regression model
        '''
        self.BayReg.setXy(X,y)
        self.BayReg.fit(X,y)
        
    def update(self,):
        '''
        Sample alpha from Bayesian regression, solve SDP, return cancidate x & (noisy) oracle function value y
        '''
        alpha_t = self.BayReg.sampleAlpha()
        
        # SDP update
        if(self.variant=='SDP'):
            self.sdp = SDP(alpha=alpha_t, lambd=self.lambd, pen_ord=self.pen_ord, mode=self.mode)
            x_new, y_new_hat = self.sdp.run()
            y_new = self.oracle.f(x_new)
        elif(self.variant=='SA'):
            pass
        else:
            assert False, "Not implemented"
        
        # update model
        self.BayReg.add(x_new, y_new)
        
        return x_new, y_new
    
class RandomSearch():
    d_MAX = 20
    
    def __init__(self, oracle:Oracle, d:int, seed:int=0):
        assert isinstance(d, int) and 0<d<self.d_MAX, f"Dimension `d` must be non-negative integer smaller than {self.d_MAX}"
        
        self.oracle = oracle
        self.d = d
        self.seed = seed
        
        np.random.seed(self.seed)
    
    def update(self,):
        '''
        Sample random x (& noisy oracle function value) y
        '''
        x_new = np.random.binomial(n=1, p=0.5, size=self.d)
        y_new = self.oracle.f(x=x_new)
        return x_new, y_new
    