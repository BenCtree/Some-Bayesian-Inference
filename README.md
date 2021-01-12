# Some-Bayesian-Inference

A couple of implementations in Matlab of some Bayesian estimation algorithms.

The poisson_estimation.m file contains implementations of random walk Metropolis-Hastings MCMC and Gaussian Variational Bayes
for estimating the beta parameter vector in a Poisson regression model. We simulate a data set using a known beta vector
and then run the algos on the simulated data in order to see how close we get to the known beta vector! Unfortunately GVB
doesn't work very well, but here it is anyway. However, MCMC works nicely!

The GARCH_MCMC.m file contains an implementation of random walk M-H MCMC to estimate the alpha, beta and omega parameters in a
GARCH(1,1) time series model. It estimates the parameters pretty well, coming close to the MLE estimate! The alpha_dist.m, beta_dist.m, omega_dist.m and sigsq_t_nonrec.m files are helper functions that work with GARCH_MCMC.m and Price_History_Commonwealth_Bank.xlsx is the time series data used.
