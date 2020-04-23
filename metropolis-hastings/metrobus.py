import matplotlib.pyplot as plt
import scipy.stats as st
import matplotlib as mpl
import seaborn as sb
import pandas as pd
import numpy as np

def target(lik, prior, theta):
    return lik(theta)

def metropolissampler(niters, lik, prior, theta,  sigma):
    accepted = 0
    samples  = [] 
    likeli   = []
    
    samples.append(theta)
    likeli.append(target(lik, prior, theta))
    
    for i in range(niters):
        theta_p = theta + sigma* np.random.normal(0.,1., len(theta)) 
        likeratio=np.exp(target(lik, prior, theta_p) - \
                            target(lik, prior, theta))        
        rho = np.minimum(1, likeratio)
        
        
        if rho > np.random.uniform():
            theta = theta_p
            accepted += 1
            
            
        samples.append(theta)
        likeli.append(target(lik, prior, theta))
    print 'Aceptance rate', 1.0*accepted/niters
    return samples