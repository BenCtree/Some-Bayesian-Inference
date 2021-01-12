% MCMC for GARCH Model - Estimating untransformed parameters
% No stationarity enforced
clear all
rng(101)

% Read in data
data = xlsread('Price_History_Commonwealth_Bank.xlsx');
% Prices column
prices = data(:,1);

% Log returns process {y_t}
returns = zeros(length(prices)-1,1);
for i=1:length(prices)-1
    returns(i) = log(prices(i+1)/prices(i));
end

% Set conditional variance at time t=1
sigsq_1 = var(returns);
% Mean of returns process
mu = mean(returns);

% Log-likelihood for one time period t
log_likelihood_t = @(a, b, w, sigsq_t_val, t) (-1/2)*(log(sigsq_t_val) + ((returns(t+1) - mu)^2 / sigsq_t_val));
%log_likelihood_t = @(a, b, w, sigsq_t_val, t) (-1/2)*(log(sigsq_t_val) + (returns(t+1)^2 / sigsq_t_val));

% Set up Markov Chain
N_iter = 20000; % number of interations 
N_burnin = 20000; % number of burnins 
N = N_iter+N_burnin; % total number of MCMC iterations 

markov_chain = zeros(N,3); % create a matrix to store the chain

% Set initial beta guesses using beta distributions as prior for alpha & beta
% And constant as prior for omega
markov_chain(1,1) = betarnd(1.5,10); % Alpha prior
markov_chain(1,2) = betarnd(10,1.5); % Beta prior
markov_chain(1,3) = 1; % Omega prior

% Set covariance matrix to generate epsilon during RWMH algo
Sigma = 0.003*eye(3);

% Prior
prior = @(a,b,w) alpha_dist(a) * beta_dist(b) * omega_dist(w);

% RWMH Algorighm
i = 1;
while i<N
    i
    epsilon = mvnrnd(zeros(3,1),Sigma);
    proposal = markov_chain(i,:)+epsilon;
    a_prop = proposal(1);
    b_prop = proposal(2);
    w_prop = proposal(3);
    a_mcmc = markov_chain(i,1);
    b_mcmc = markov_chain(i,2);
    w_mcmc = markov_chain(i,3);
    sigsqs_prop = sigsq_t_nonrec(a_prop, b_prop, w_prop, length(returns), returns, sigsq_1);
    prop_log_lik_ts = zeros(length(sigsqs_prop)-1,1);
    mc_log_lik_ts = zeros(length(sigsqs_prop)-1,1);
    for t = 1:length(sigsqs_prop)-1
        sigsq_t_val = sigsqs_prop(t);
        prop_log_lik_ts(t) = log_likelihood_t(a_prop, b_prop, w_prop, sigsq_t_val, t);
        mc_log_lik_ts(t) = log_likelihood_t(a_mcmc, b_mcmc, w_mcmc, sigsq_t_val, t);
    end
    prop_kernel = log(prior(a_prop, b_prop, w_prop)) - ((length(returns)-1)*log(2*pi)/2 + sum(prop_log_lik_ts));
    mcmc_kernel = log(prior(a_mcmc, b_mcmc, w_mcmc)) - ((length(returns)-1)*log(2*pi)/2 + sum(mc_log_lik_ts));
    auxiliary = prop_kernel - mcmc_kernel;
    alpha = min(exp(auxiliary),1);
    u = rand;
    if u<alpha
        markov_chain(i+1,:) = proposal;
    else
        markov_chain(i+1,:) = markov_chain(i,:);
    end
    i = i+1;
end

% Trace plot for each parameter
hold on
plot(markov_chain(:,1))
%title('Trace Plot \alpha')
plot(markov_chain(:,2))
%title('Trace Plot \beta')
plot(markov_chain(:,3))
%title('Trace Plot \omega')
legend('alpha', 'beta', 'omega')
title('Trace Plots')
hold off

% Remove burn-in period
markov_chain_noburn = markov_chain(N_burnin+1:N,:);

% Trace plot for each parameter with no burnin period
hold on
plot(markov_chain_noburn(:,1))
plot(markov_chain_noburn(:,2))
plot(markov_chain_noburn(:,3))
legend('alpha', 'beta', 'omega')
title('Trace Plots Without Burnin Period')
hold off

% Posterior Means
pm_alpha = mean(markov_chain_noburn(:,1));
pm_beta = mean(markov_chain_noburn(:,2));
pm_omega = mean(markov_chain_noburn(:,3));

post_means = [pm_alpha, pm_beta, pm_omega]

% Posterior Variances
pv_alpha = var(markov_chain_noburn(:,1));
pv_beta = var(markov_chain_noburn(:,2));
pv_omega = var(markov_chain_noburn(:,3));

post_variances = [pv_alpha, pv_beta, pv_omega]