% Calculate log-posterior for DPM-IRL
%
function [logPost, Qcells, logDPprior, logLLH, logPrior] = ...
    calDPMLogPost(trajSet, belongTo, wSet, policySet, mdp, opts)

logDPprior = vpa(log(calClustAssignProb(belongTo, opts.alpha)));
logDPprior
logLLH     = 0;
logPrior   = 0;

% save optimal Q values
Qcells = cell(max(belongTo), 1);

for k = 1:max(belongTo)
    w = wSet(:, k);
    if isempty(policySet)
        mdp    = convertW2RSepsis(w, mdp);
        policy = policyIteration(mdp);
    else
        policy = policySet(:, k);
    end

    X = getTrajInfo(trajSet(belongTo == k, :, :), mdp);
    [llh, NBQ]   = calLogLLH(w, X, policy, [], [], mdp, opts);
    prior = calLogPrior(w, opts);
    logLLH   = logLLH + llh;
    logPrior = logPrior + prior;
    Qcells{k} = NBQ;    % total log likelihoods of the observations
end
logLLH
logPrior
logPost = logDPprior + logLLH + logPrior;

end