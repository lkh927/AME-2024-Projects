import numpy as np 
from scipy.stats import norm

name = 'Tobit'

def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[0] is beta, 
        theta[-2] is sigma (not squared), and
        theta[-1] is mubut 
    '''
    N,K = x.shape 
    b_ols = np.linalg.solve(x.T@x, x.T@y)
    res = y - x@b_ols 
    mu = np.mean(res)
    sig2hat = 1./(N-K) * np.dot(res, res)
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    theta0 = np.append(theta0, mu)
    return theta0

def q(theta, y, x): 
    return loglikelihood(theta, y, x)

def loglikelihood(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-2] 
    sig = np.abs(theta[1]) # take abs() to ensure positivity 
    mu = theta[2]
    N,K = x.shape

    xb_s = (x@b + mu) / sig
    Phi = norm.cdf(xb_s)

    u_s = (y - (mu + x@b)) / sig
    phi = norm.pdf(u_s) / sig

    # avoid taking log of zero
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)
    phi = np.clip(phi, 1e-8, 1.-1e-8)

    # loglikelihood function 
    ll = (y == 0.0) * np.log(Phi) + (y < 0) * np.log(phi)

    return ll

def mills_ratio(z): 
    return norm.pdf(z) / norm.cdf(z)

# def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
        E_neg: E(y|x, y<0) 
    '''
    b = theta[:-1]
    s = theta[-1]
    #m = theta[-1]
    xb = x@b 
    E = ((1 - norm.cdf((m + xb)/ s))(m+xb) - s * norm.pdf((m+xb)/ s))
    Eneg = xb - s * mills_ratio(-xb/s)
    return E, Eneg


