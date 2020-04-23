import emcee
import numpy as np
import os
import sys
import emcee


def theory(x, m, c):
    """
    A straight line model: y = m*x + c

    Parameters:

        x (list): a set of abscissa points at which the model is defined
        m (float): the gradient of the line
        c (float): the y-intercept of the line
    """

    return m*x+c


# ##########create some data#######################################
# set the true values of the model parameters for creating the data

m = 3.5  # gradient of the line
c = 1.2  # y-intercept of the line

# set the "predictor variable"/abscissa
M = 100
xmin = 0.
xmax = 10.

stepsize = (xmax-xmin)/M

x = np.arange(xmin, xmax, stepsize)
# create the data - the model plus Gaussian noise
sigma = 0.5
data = theory(x, m, c) + sigma*np.random.randn(M)

nDims = 2
# bounds = [bounds_par1, bounds_par2, ...]
# bounds is list of lists
#m c
bounds = [[0.0,5.0], [-10.0,10.0]]
##########

def logposterior(theta):
    """
    The natural logarithm of the joint posterior.
    
    Args:
        theta (tuple): a sample containing individual parameter values
        data (list): the set of data/observations
        sigma (float): the standard deviation of the data points
        x (list): the abscissa values at which the data/model is defined
    """
    
    lp = logprior(theta) # get the prior
    
    # if the prior is not finite return a probability of zero (log probability of -inf)
    if not np.isfinite(lp):
        return -np.inf
    
    # return the likeihood times the prior (log likelihood plus the log prior)
    return lp + loglikelihood(theta)


def loglikelihood(theta):
    
    m, c = theta
    
    # evaluate the model (assumes that the straight_line model is defined as above)
    th = theory(x, m, c)
    
    # return the log likelihood
    return -0.5*np.sum(((th - data)/sigma)**2)

# bounds = [[], []]

def logprior(theta):
    """
    The natural logarithm of the prior probability.
    
    Args:
        theta (tuple): a sample containing individual parameter values
    
    Note:
        We can ignore the normalisations of the prior here.
    """
    
    lp = 0.
    
    # unpack the model parameters from the tuple
    # m, c = theta
    
    # uniform prior on c
    cmin = -10. # lower range of prior
    cmax = 10.  # upper range of prior
    
    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range 
    for i in range(len(bounds)):
    	if bounds[i][0] < theta[i] < bounds[i][1]:
    		flag = True
    	else:
    		flag = False
    #if bounds[0][0] < theta[0] < bounds[0][1] and bounds[1][0] < theta[1] < bounds[1][1]:
    #if bounds[0][0] < theta[0] < bounds[0][1] and -bounds[1][0] < theta[1] < bounds[1][1:
    if flag == True:
    	return 0.0

    return -np.inf
    
    # Gaussian prior on m
    # mmu = 0.     # mean of the Gaussian prior
    # msigma = 10. # standard deviation of the Gaussian prior
    # lp -= 0.5*((m - mmu)/msigma)**2
    # lp += 0. if 0 < m < 10 else -np.inf

    # return lp

Nens = 100   # number of ensemble points

mmu = 0.     # mean of the Gaussian prior
msigma = 10. # standard deviation of the Gaussian prior

# mini = np.random.normal(mmu, msigma, Nens) # initial m points
mini = np.random.uniform(0, 5, Nens) # initial m points
cmin = -10.  # lower range of prior
cmax = 10.   # upper range of prior

cini = np.random.uniform(cmin, cmax, Nens) # initial c points

inisamples = np.array([mini, cini]).T # initial samples

ndims = inisamples.shape[1] # number of parameters/dimensions

Nburnin = 500   # number of burn-in samples
Nsamples = 500  # number of final posterior samples

# set additional args for the posterior (the data, the noise std. dev., and the abscissa)
#argslist = (data, msigma, x)

# set up the sampler
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior)

# pass the initial samples and total number of samples required
sampler.run_mcmc(inisamples, Nsamples+Nburnin);

# extract the samples (removing the burn-in)
postsamples = sampler.chain[:, Nburnin:, :].reshape((-1, ndims))

# plot posterior samples (if corner.py is installed)
import corner # import corner.py

print('Number of posterior samples is {}'.format(postsamples.shape[0]))

fig = corner.corner(postsamples, labels=[r"$m$", r"$c$"])
fig.savefig('emcee.png')