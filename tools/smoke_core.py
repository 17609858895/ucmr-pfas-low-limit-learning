"""Small deterministic numerical checks; no model training and no raw data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import numpy as np
from censored_models import Dist, Mixture, censored_loglik, mixture_loglik, fit_stack_weights

def main():
    mu = np.array([.3,.7,1.1,1.7])
    normal = Dist(mu,.4,pi=np.array([.1,.4,.7,.95]))
    logistic = Dist(mu+.1,.3,pi=np.array([.2,.5,.8,.99]),family='logistic')
    thresholds = np.log10(np.array([4.,6.,8.,10.,15.,20.]))
    p = np.column_stack([normal.sf(t) for t in thresholds])
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    assert (np.diff(p,axis=1)<=0).all()
    assert np.allclose(normal.cdf(thresholds[0])+normal.sf(thresholds[0]),1)
    lo = np.array([-np.inf,.7,-np.inf,1.7]); hi=np.array([.6,.7,.6,1.7])
    weights = fit_stack_weights([normal,logistic],lo,hi)
    assert np.isclose(weights.sum(),1) and (weights>=0).all()
    mixture = Mixture([normal,logistic],weights)
    assert np.isfinite(censored_loglik(normal,lo,hi)).all()
    assert np.isfinite(mixture_loglik(mixture,lo,hi)).all()
    assert np.allclose(mixture.sf(thresholds[0]),weights[0]*normal.sf(thresholds[0])+weights[1]*logistic.sf(thresholds[0]))
    print('PASS: ordered probabilities, distribution complement, censored likelihood, simplex pooling.')

if __name__=='__main__':
    main()
