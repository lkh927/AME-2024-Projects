import numpy as np 
from scipy.stats import norm

name = 'Tobit'

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity 
    mu = 1
    N,K = x.shape

    xb_s = x@b + mu / sig
    Phi = norm.cdf(xb_s)

    u_s = (y - (mu + x@b)) / sig
    phi = norm.pdf(u_s) / sig

    # avoid taking log of zero
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)

    # loglikelihood function 
    ll = (y == 0.0) * np.log(Phi) + (y < 0) * np.log(phi)

    return ll

def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N,K = x.shape 
    b_ols = np.linalg.solve(x.T@x, x.T@y)
    res = y - x@b_ols 
    sig2hat = 1./(N-K) * np.dot(res, res)
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

def mills_ratio(z): 
    return norm.pdf(z) / norm.cdf(z)

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
        E_neg: E(y|x, y<0) 
    '''
    b = theta[:-1]
    s = theta[-1]
    xb = x@b 
    E = xb * (1 - norm.cdf(xb/s)) + s * norm.pdf(xb/s)
    Eneg = xb - s * mills_ratio(-xb/s)
    return E, Eneg


