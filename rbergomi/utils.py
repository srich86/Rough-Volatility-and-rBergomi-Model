import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V, o = 'call'):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

def bs_vega(S, K, T, r, sigma):
    # Spot price
    # K: Strike price
    # T: time to maturity
    # r: interest rate (1=100%)
    # sigma: volatility of underlying asset
    
    d1 = (np.log(S/K) + (r + 0.5 * np.power(sigma, 2)) * T) / sigma *np.sqrt(T)
    vg = S * norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    vega = np.maximum(vg, 1e-19)
    return vega

def call_price(sigma, S, K, r, T):
    d1 = 1 / (sigma * np.sqrt(T)) * ( np.log(S/K) + (r + np.power(sigma,2)/2) * T)
    d2 = d1 - sigma * np.sqrt(T)
    C = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
    return C


def find_vol(target_value, S, K, T, r):
    # target value: price of the option contract
    MAX_iterations = 10000
    prec = 1.0e-8
    sigma = 0.5
    for i in range(0, MAX_iterations):
        price = call_price(sigma, S, K, r, T)
        diff = target_value - price # the root
        if abs(diff) < prec:
            return sigma
        vega = bs_vega(S, K, T, r, sigma)
        sigma = sigma + diff/vega # f(x) / f'(x)
        if sigma > 10 or sigma < 0:
            sigma=0.5
        
    return sigma
    
def bsinv(P, F, K, t, o = 'call'):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    
    s = brentq(error, 1e-19, 1e+9)

    return s
    
#def bsinv(P, F, K, t, r, o = 'call'):
#    """
#    Returns implied Black vol from given call price, forward, strike and time
#    to maturity.
#    """
#    # Set appropriate weight for option token o
#    w = 1
#    if o == 'put':
#        w = -1
#    elif o == 'otm':
#        w = 2 * (K > 1.0) - 1
#
#    # Ensure at least instrinsic value
#    P = np.maximum(P, np.maximum(w * (F - K), 0))
#
#    def error(s):
#        return bs(F, K, s**2 * t, o) - P
#    
#    try:
#        s = brentq(error, 1e-19, 1e+9)
#    except Exception as e:
#        print(e)
#        try:
#            s = find_vol(P, F, K, t, r)
#        except Exception as e:
#            print(e)
#            s = 1e-19
#    if s == 1e-19:
#        try:
#            s = find_vol(P, F, K, t, r)
#        except Exception as e:
#            print(e)
#            s = 1e-19
#    return s
