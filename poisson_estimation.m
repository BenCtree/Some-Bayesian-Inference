% Using MCMC and Cholesky GVB to estimate parameters
% of a Poisson regression model on simulated data to determine accuracy
% (how close each method gets to a known beta parameter vector)

% Setup

% Fix the random seed so that results are reproducible.
rng(0)

% Generate  the  covariates
% xij~U(0,1), i=1,...,n=100; j=1,...,4.
n = 100;
col1 = ones(n,1);
X4 = rand(n,4);
X = [col1 X4];

% Set actual beta
beta = [1,-0.2,0.4,0.1,-0.7];

% Generate a data set of n=100 observations
% according to the given model.

mu_vect = zeros(n,1);

for i=1:n
    mu_vect(i) = exp(sum(beta.*X(i, 1:5)));
end

y_vect = zeros(n,1);

for i=1:n
    y_vect(i) = poissrnd(mu_vect(i));
end

% Perform Bayesian inference on beta using MCMC:
% Use a Metropolis-Hastings algorithm
% to estimate the posterior distribution of beta

% Random Walk Metropolis-Hastings

N_iter = 20000; % number of interations 
N_burnin = 10000; % number of burnins 
N = N_iter+N_burnin; % total number of MCMC iterations 

%step = 0.1;
betaTxi = @(beta_est, i) sum(beta_est.*X(i, 1:5));
log_likelihood_i = @(beta_est, i) y_vect(i)*betaTxi(beta_est, i) - exp(betaTxi(beta_est,i)) - log(factorial(y_vect(i)));
log_likelihood = @(beta_est) 0
for i=1:n
    log_likelihood = @(beta_est) log_likelihood(beta_est) + log_likelihood_i(beta_est, i)
end
% Mean vector and cov matrix for multivariate normal dist
% which is prior for beta.
beta_mu = zeros(1,5);
beta_Sigma = 100.*eye(5);
% Evaluate pdf of multivariate normal dist
% to get probabilities for each beta component
% function to compute the kernel k(beta)

markov_chain = zeros(N,5); % create a matrix to store the chain

% Set initial beta guess using multivariate normal prior
markov_chain(1,:) = mvnrnd(beta_mu,beta_Sigma);

% Set inverse of Fisher Info matrix as sigma
% for epsilon multivariate normal dist
Sigma = eye(size(X,2))/(X'*X);
i = 1;
while i<N
    epsilon = mvnrnd(zeros(5,1),Sigma);
    proposal = markov_chain(i,:)+epsilon;
    %auxiliary = k(proposal)-k(markov_chain(i,:));
    auxiliary = log_likelihood(proposal)-log_likelihood(markov_chain(i,:));
    alpha = min(exp(auxiliary),1);
    u = rand;
    if u<alpha
        markov_chain(i+1,:) = proposal;
    else
        markov_chain(i+1,:) = markov_chain(i,:);
    end
    i = i+1;
end

% Remove burn-in period
markov_chain_noburn = markov_chain(N_burnin+1:N,:);

% Trace plot for each beta component
hold on
plot(markov_chain_noburn(:,1))
title('Trace Plot \beta_0')
plot(markov_chain_noburn(:,2))
title('Trace Plot \beta_1')
plot(markov_chain_noburn(:,3))
title('Trace Plot \beta_2')
plot(markov_chain_noburn(:,4))
title('Trace Plot \beta_3')
plot(markov_chain_noburn(:,5))
title('Trace Plot \beta_4')
hold off

% Posterior Mean Estimates for each beta
post_mean_est_beta1 = mean(markov_chain_noburn(:,1));
post_mean_est_beta2 = mean(markov_chain_noburn(:,2));
post_mean_est_beta3 = mean(markov_chain_noburn(:,3));
post_mean_est_beta4 = mean(markov_chain_noburn(:,4));
post_mean_est_beta5 = mean(markov_chain_noburn(:,5));

post_mean_betas_mcmc = [post_mean_est_beta1, post_mean_est_beta2, post_mean_est_beta3, post_mean_est_beta4, post_mean_est_beta5];

% Distance from actual beta
dist_mcmc = norm(beta - post_mean_betas_mcmc)

% Posterior Standard Deviation Estimates for each beta
post_sd_est_beta1 = sqrt(var(markov_chain_noburn(:,1)));
post_sd_est_beta2 = sqrt(var(markov_chain_noburn(:,2)));
post_sd_est_beta3 = sqrt(var(markov_chain_noburn(:,3)));
post_sd_est_beta4 = sqrt(var(markov_chain_noburn(:,4)));
post_sd_est_beta5 = sqrt(var(markov_chain_noburn(:,5)));

post_sd_betas_mcmc = [post_sd_est_beta1, post_sd_est_beta2, post_sd_est_beta3, post_sd_est_beta4, post_sd_est_beta5];

% Estimate predictive mean of new observation x*

x_star = [1,1.8339,-2.2588,0.8622,0.3188];
mu_star_vect = zeros(N_iter, 1);
for i=1:N_iter
    mu_star_vect(i) = exp(sum(markov_chain_noburn(i,:).*x_star));
end
y_star_vect = zeros(N_iter, 1);
for i=1:N_iter
    y_star_vect(i) = poissrnd(mu_star_vect(i));
end

% Posterior predictive distribution
hist(y_star_vect)
% Posterior predictive mean
post_pred_mean_mcmc = mean(y_star_vect)

% Cholesky GVB

% Initialise variables

%lambda_mu0 = zeros(5,1);
lambda_mu0 = (X'*X)\(X'*y_vect);
%lambda_Sigma = eye(size(X,2))/(X'*X); % Inverse Fisher Info gives mean_betas = 1.6868   -1.2486    0.4852   -0.7828   -2.1003
%lambda_Sigma = 10*eye(5); % LB_bar plot not good Gives distance = 1.8822, mean_betas = 2.5144   -0.4576    1.2841    0.0387   -1.3305
lambda_Sigma = eye(5); % Gives distance = 2.0719, mean_betas = 1.6909   -1.2447    0.4895   -0.7779   -2.0948
%lambda_Sigma = ones(5); % Gives mean_betas = 2.4196   -0.5156    1.2183   -0.0507   -1.3714
lambda_Sigma = tril(lambda_Sigma);
vech_Sigma = lambda_Sigma(tril(true(size(lambda_Sigma))));
lambda = [lambda_mu0; vech_Sigma]; % Initial vector of variational parameters
d = length(lambda); % Length of lambda
S = 2000; % number of Monte Carlo samples
beta1_adap_weight = 0.9; % adaptive learning weight
beta2_adap_weight = 0.9; % adaptive learning weight
eps0 = 0.005; % Fixed learning rate
max_iter = 200;
patience_max = 10;
tau_threshold = max_iter/2; % Learning rate threshold
t_w = 50; % Rolling window size

eps_mu = zeros(5,1);
eps_sig = eye(5);
eps_s = mvnrnd(eps_mu, eps_sig, S);
grad_h_lambda = zeros(S,5);
vech_ghlte = zeros(S,15);
grad_log_likelihood_i = @(beta_est, i) (y_vect(i)*X(i,:) - X(i,:)*exp(betaTxi(beta_est,i)))';
grad_log_likelihood = @(beta_est) 0;
for i=1:n
    grad_log_likelihood = @(beta_est) grad_log_likelihood(beta_est) + grad_log_likelihood_i(beta_est, i);
end
parfor s = 1:S
    beta_s = lambda_mu0' + eps_s(s,:)*lambda_Sigma;
    grad_h_lambda(s,:) = -beta_s/beta_Sigma + grad_log_likelihood(beta_s)' - (beta_s - lambda_mu0')/lambda_Sigma;
    grad_h_lambda_times_eps = (grad_h_lambda(s,:)'*eps_s(s,:));
    % transp_ghlte = grad_h_lambda_times_eps';
    vech_ghlte(s,:) = (grad_h_lambda_times_eps(tril(true(size(grad_h_lambda_times_eps)))))';
end

% Estimate of lower bound gradient
grad_LB = [mean(grad_h_lambda)'; mean(vech_ghlte)'];

g_adaptive = grad_LB;
v_adaptive = g_adaptive.^2; 
g_bar_adaptive = g_adaptive;
v_bar_adaptive = v_adaptive;

iter = 1;
stop = false;
LB = 0;
LB_bar = 0;
patience = 0;
while ~stop
    iter
    lambda_mu = lambda(1:5);
    lambda_Sigma = zeros(5);
    lambda_Sigma(:,1) = lambda(6:10);
    lambda_Sigma(2:5,2) = lambda(11:14);
    lambda_Sigma(3:5,3) = lambda(15:17);
    lambda_Sigma(4:5,4) = lambda(18:19);
    lambda_Sigma(5,5) = lambda(20);
    h_lambda = zeros(S,1); % function h_lambda
    log_prior = @(beta_est) ((-5/2)*log(2*pi) - (1/2)*log(det(100*eye(5))) - (1/2)*beta_est*(inv(100*eye(5)))*beta_est');
    log_q_lambda = @(beta_est) ((-5/2)*log(2*pi) - (1/2)*log(det(lambda_Sigma)) - (1/2)*(beta_est-lambda_mu')*(inv(lambda_Sigma))*(beta_est' - lambda_mu));
    eps_mu = zeros(5,1);
    eps_sig = eye(5);
    eps_s = mvnrnd(eps_mu, eps_sig, S);
    grad_h_lambda = zeros(S,5);
    vech_ghlte = zeros(S,15);
    parfor s = 1:S
        beta_s = lambda_mu0' + eps_s(s,:)*lambda_Sigma;
        h_lambda(s) = log_prior(beta_s) + log_likelihood(beta_s) - log_q_lambda(beta_s);
        grad_h_lambda(s,:) = -beta_s/beta_Sigma + grad_log_likelihood(beta_s)' - (beta_s - lambda_mu0')/lambda_Sigma;
        grad_h_lambda_times_eps = (grad_h_lambda(s,:)'*eps_s(s,:))';
        % transp_ghlte = grad_h_lambda_times_eps';
        vech_ghlte(s,:) = (grad_h_lambda_times_eps(tril(true(size(grad_h_lambda_times_eps)))))';
    end
    
    grad_LB = [mean(grad_h_lambda)'; mean(vech_ghlte)'];
    
    g_adaptive = grad_LB;
    v_adaptive = g_adaptive.^2; 
    g_bar_adaptive = beta1_adap_weight*g_bar_adaptive+(1-beta1_adap_weight)*g_adaptive;
    v_bar_adaptive = beta2_adap_weight*v_bar_adaptive+(1-beta2_adap_weight)*v_adaptive;
    
    if iter>=tau_threshold
        stepsize = eps0*tau_threshold/iter;
    else
        stepsize = eps0;
    end
    
    vech_Sigma = lambda_Sigma(tril(true(size(lambda_Sigma))));
    lambda = [lambda_mu; vech_Sigma];
    lambda = lambda+stepsize*g_bar_adaptive./sqrt(v_bar_adaptive);
    
    LB(iter) = mean(h_lambda);
    
    if iter>=t_w
        LB_bar(iter-t_w+1) = mean(LB(iter-t_w+1:iter));
        LB_bar(iter-t_w+1)
    end
       
    if (iter>t_w)
        if (LB_bar(iter-t_w+1)>=max(LB_bar))
            lambda_best = lambda;
            patience = 0;
        else
            patience = patience+1;
        end
    end
    
    if (patience>patience_max)||(iter>max_iter) stop = true; end 
        
    iter = iter+1;
 
end

lambda = lambda_best;

% Convert lambda vector back to mu vector and lower triangular matrix
% Obtain sigma matrix from lower triangular matrix

lambda_mu_final = lambda(1:5);

L = zeros(5);
L(:,1) = lambda(6:10);
L(2:5,2) = lambda(11:14);
L(3:5,3) = lambda(15:17);
L(4:5,4) = lambda(18:19);
L(5,5) = lambda(20);
lambda_sigma_final = L*L';

% Trace plot of moving average of lower bounds
%plot(LB_bar)
plot(LB_bar)
title('Trace Plot Lower Bound Moving Average')

% Posterior means for each beta component

% Generate 20,000 beta vectors from multivariate normal dist
% using mu and sigma derived from GVB
beta_ests = mvnrnd(lambda_mu_final',lambda_sigma_final,N_iter);

post_mean_betas_vb = [mean(beta_ests(:,1)), mean(beta_ests(:,2)), mean(beta_ests(:,3)), mean(beta_ests(:,4)), mean(beta_ests(:,5))]

% Distance from actual beta
dist_vb = norm(beta - post_mean_betas_vb)

% Posterior standard deviation for each beta component

post_sd_betas_vb = [std(beta_ests(:,1)), std(beta_ests(:,2)), std(beta_ests(:,3)), std(beta_ests(:,4)), std(beta_ests(:,5))]

% Given  a  future  subject  with  covariates x*=  (1.8339,-2.2588,0.8622,0.3188),
% estimate the predictive mean E(y*|x*,D) based on your VB approximation.

x_star = [1,1.8339,-2.2588,0.8622,0.3188];
mu_star_vect_vb = zeros(N_iter, 1);
for i=1:N_iter
    mu_star_vect_vb(i) = exp(sum(beta_ests(i,:).*x_star));
end
y_star_vect_vb = zeros(N_iter, 1);
for i=1:N_iter
    y_star_vect_vb(i) = poissrnd(mu_star_vect_vb(i));
end

% Posterior predictive distribution of y
hist(y_star_vect_vb)
% Posterior predictive mean
post_pred_mean_vb = mean(y_star_vect_vb)
