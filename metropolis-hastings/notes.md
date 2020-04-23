This Metropolis-Hastings implementation follows the recommendations of:

	- Hogg, D. W., & Foreman-Mackey, D. (2018). Data analysis recipes: Using markov chain monte carlo. The Astrophysical Journal Supplement Series, 236(1), 11.

In this notes, I write some of these recommendations.

	- Autocorrelation time is the most important test for convergence

	- MCMC methods are samplers not optimizers.

	- Metropolis-Hastings requires two inputs: 
		1) A density function f(theta), it can be prior times likelihood.
		2) A PDF q(theta'|theta) that can deliver samples.

	- Why MH works?:
		1) The Markov process delivers a unique stationary distribution. 
		2) The stationary distribution is proportional to the density function f(q).

	

