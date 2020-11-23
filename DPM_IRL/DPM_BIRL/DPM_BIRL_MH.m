% Metropolis-Hastings algorithm for IRL with multiple experts
%
function [sol, logPost, hst] = DPM_BIRL_MH(trajSet, mdp, opts)

fprintf('- DPM-BIRL using MH\n');

useExisting = 0
% build distribution for 1d reward value
opts = getRewardDist(opts);

% cache Pr((s,a)) for weight matrix
Q_cache = java.util.HashMap;

% trajectory data
tr.set = trajSet;          % trajectories
tr.n   = size(tr.set, 1);  % number of trajectories
tr.cnt = cell(tr.n, 1);    % information of trajectories
for i = 1:tr.n
    info = getTrajInfo(tr.set(i, :, :), mdp);
    tr.cnt{i} = info.cnt;
end
startT = tic;
% initialize cluster
if useExisting
    load('./sol_800.mat');
    cl.b = sol.belongTo
    cl.w = sol.weight
    cl.v = sol.value
    cl.p = sol.policy
else
    cl = init(tr, mdp, opts, Q_cache);
    cl = relabelClustAssign(cl);
end
elapsedT = toc(startT);

[pr, Qcells] = calDPMLogPost(tr.set, cl.b, cl.w, cl.p, mdp, opts);

fprintf('+    init : %10.2f %10s %8.2f sec ', pr, '', elapsedT);
printClustAssign(cl.b);
maxCl.logPost = -inf;
[maxCl, hst, ~] = saveHist(cl, pr, maxCl, [], 1, elapsedT, Qcells);


for iter = 1:opts.maxIters
    fprintf('iterations: %d', iter)
    tempT_1 = tic;
    for m = randperm(tr.n)      % random permutation of ordering of trajectories
        cl = updateClustAssign(m, tr.cnt, cl, mdp, opts, Q_cache);
    end
    elapsedT = toc(tempT_1)
    fprintf('time for cluster assign: %8.2f',elapsedT)
    cl = relabelClustAssign(cl);
    tempT_2 = tic;
    randperm(max(cl.b))

    tot = 0;
    acceptance = 0;
    for k = randperm(max(cl.b)) % random permutation of ordering of clusters
        [cl,tot1,accept1] = updateWeight(k, tr.cnt, cl, mdp, opts, tot, acceptance, Q_cache);
        tot = tot + tot1;
        acceptance = acceptance + accept1;
    end
    elapsedT = toc(tempT_2);
    fprintf('time for reward update: %8.2f', elapsedT)

    elapsedT = toc(startT);
    [pr, Qcells] = calDPMLogPost(tr.set, cl.b, cl.w, cl.p, mdp, opts);
    [maxCl, hst, bUpdate] = saveHist(cl, pr, maxCl, hst, iter + 1, elapsedT, Qcells);

    if (mod(iter,50) == 0)
        fprintf('saving maxCl \n');
        sol = maxCl;
        save('test.mat','sol')
    end

    fprintf('weight acceptance: %8.2f', tot / acceptance);
    tot = 0;
    acceptance = 0;
    if iter == 1 || mod(iter, 100) == 0 || bUpdate || elapsedT > opts.maxTime
        fprintf('+ %4d-th : %10.2f %10.2f %8.2f sec ', ...
            iter, pr, maxCl.logPost, elapsedT);
        printClustAssign(maxCl.belongTo);
    end
    %if elapsedT > opts.maxTime
        %fprintf('# time out\n');
        %break;
    %end

    % reset unused Q values in cache
    Q_cache_reset = java.util.HashMap;
    for k = 1:size(unique(cl.b), 1)
        weight = cl.w(:,k);
        NBQ = Q_cache.get(key_enc(weight'));
        Q_cache_reset.put(key_enc(weight'), NBQ);
    end
    Q_cache = Q_cache_reset;

end
sol     = maxCl;
logPost = maxCl.logPost;
save('sol.mat','sol')
fprintf('\n\n');

end


function [maxCl, hst, bUpdate] = saveHist(cl, pr, maxCl, hst, i, elapsedT, Qcells)

bUpdate = false;
if maxCl.logPost < pr
    maxCl.belongTo = cl.b;
    maxCl.weight   = cl.w;
    maxCl.policy   = cl.p;
    maxCl.value    = cl.v;
    maxCl.logPost  = pr;
    maxCl.Qcells = Qcells;
    bUpdate = true;
end
hst.cl{i}      = cl;
hst.logPost(i) = pr;
hst.maxLogPost(i) = maxCl.logPost;
hst.elapsedT(i)   = elapsedT;

end


% Perform Metropolis-Hastings update for cluster assignment of m-th
% trajectory
function cl = updateClustAssign(m, trajCnt, cl, mdp, opts, Q_cache)

% if there is an existing cluster already, cl.w fixed.
% but probability of assignment into a cluster is proportional to the size of that cluster.
% if try to parallerize for loop then the proportionality to cluster size is destroyed.


for iiter = 1:opts.clIters;
    c = cl.b(m);    % m is the index. c = which cluster
    w = cl.w(:, c); % which weight
    v = cl.v(:, c);

    N = max(cl.b);
    prior = zeros(N + 1, 1);
    %startT = tic;
    for k = 1:N
        % prior(k) denotes how many trajectories belong to kth cluster
        prior(k) = sum(cl.b == k);
    end
    prior(c)   = prior(c) - 1;  % take out c
    prior(end) = opts.alpha;
    %endT = toc(startT);
    %elapsedT = endT - startT;
    %fprintf('getting clusters into priors %8.10f \n', elapsedT)

    %startT = tic;
    c2 = sampleMultinomial(prior);  % sample cluster c_i based on probs. proportional to prior vector
    % endT = toc(startT);
    % elapsedT = endT - startT;
    % fprintf('sampling multinomial %8.10f \n', elapsedT)

    ratio = 0;
    if c2 > N
        [w2, p2, v2] = newWeight(mdp, opts);

        % if new weight drawn, then initialise NBQ values
        Q_vals = QfromV(v, mdp, w2);

        BQ2 = opts.eta .* Q_vals;

        BQsum2 = log(sum(exp(BQ2), 2));

        NBQ2  = bsxfun(@minus, BQ2, BQsum2);

        ratio = calAcceptRatioCluster(trajCnt{m}, NBQ2, Q_cache.get(key_enc(w')));

    else
        w2 = cl.w(:, c2);
        p2 = cl.p(:, c2);
        v2 = cl.v(:, c2);

        ratio = calAcceptRatioCluster(trajCnt{m}, Q_cache.get(key_enc(w2')), Q_cache.get(key_enc(w')));

    end
    %startT = tic;
    %ratio = calAcceptRatioCluster(trajCnt{m}, w2, v2, w, v, mdp, opts);
    %endT = toc(startT);
    %elapsedT = endT - startT;
    %fprintf('calculate ratio %8.8f \n', elapsedT)
    if rand < ratio
        cl.b(m) = c2;
        if c2 > N
            cl.w(:, c2) = w2;
            cl.p(:, c2) = p2;
            cl.v(:, c2) = v2;

            % cache Q vals
            Q_cache.put(key_enc(w2'), NBQ2)
        end
    end
end
% endT = toc(startT);
% elapsedT = endT - startT;
% fprintf('one sampling routine %8.10f \n', elapsedT)
end


% Perform Metropolis-Hastings update for weight of k-th cluster
function [cl,tot,acceptance] = updateWeight(k, trajCnt, cl, mdp, opts, tot, acceptance, Q_cache)

for iiter = 1:opts.wIters;
    % (cl.b == k) = indices which belong to the kth cluster
    trajs = trajCnt(cl.b == k);
    w = cl.w(:, k);
    v = cl.v(:, k);

    [w2, p2, v2] = newWeight(mdp, opts);

    % if the log likelihood of the new reward function is
    % higher compared to the old reward function
    ratio = calAcceptRatio(trajs, w2, v2, w, v, mdp, opts);
    % if the ratio of the likelihood of the observations under the new rewar function (w2)
    % is greater than the likelihood of the observations under the old reward function (w1)
    % then accept the new reward weight
    tot = tot + 1;
    if rand < ratio
        cl.w(:, k) = w2;
        cl.p(:, k) = p2;
        cl.v(:, k) = v2;

        % set log Pr((s,a)) for w2
        acceptance = acceptance + 1;
    end
end
% save NBQ1 at the end
Q_vals = QfromV(cl.v(:, k), mdp, cl.w(:,k));

BQ1 = opts.eta .* Q_vals;

BQsum1 = log(sum(exp(BQ1), 2));

NBQ1  = bsxfun(@minus, BQ1, BQsum1);

Q_cache.put(key_enc(cl.w(:,k)'), NBQ1);

end



% Relabel cluster assignment
function cl = relabelClustAssign(cl)

tmpId = [];
W = [];
P = [];
V = [];

bRelabel = false;
for k = 1:max(cl.b)
    if sum(cl.b == k) > 0
        W = cat(2, W, cl.w(:, k));
        P = cat(2, P, cl.p(:, k));
        V = cat(2, V, cl.v(:, k));
        tmpId   = cat(1, tmpId, [k, size(W, 2)]);
        bRelabel = true;
    end
end

if bRelabel
    B = zeros(length(cl.b), 1);
    for i = 1:size(tmpId, 1)
        B(cl.b == tmpId(i, 1)) = tmpId(i, 2);
    end
    cl.b = B;
    cl.w = W;
    cl.p = P;
    cl.v = V;
end

end


% Sample new weight and compute its policy and value
function [w, p, v] = newWeight(mdp, opts)

w      = sampleNewWeight(mdp.nFeatures, opts);
mdp    = convertW2RSepsis(w, mdp);
[p, v] = policyIteration(mdp);

end


% Initialize
function cl = init(tr, mdp, opts, Q_cache)
%
% cl.b = [];
% cl.w = [];
% cl.p = [];
% cl.v = [];
% for i = 1:tr.n
%     w = MAP_BIRL(tr.set(i, :, :), [], mdp, opts);
%     mdp    = convertW2R(w, mdp);
%     [p, v] = policyIteration(mdp);
%     cl.b(:, i) = i;
%     cl.w(:, i) = w;
%     cl.p(:, i) = p;
%     cl.v(:, i) = v;
% end

initial_cluster = 10;
cl.b = randi(initial_cluster, tr.n, 1);
%cl.b = randi(tr.n, tr.n, 1); %ones(tr.n, 1); %
%cap intial cluster size
%cl.b = cluster asssignment for each data trajectory
%cl.w = cluster weight for each data trajectory
%cl.p = cluster policy for each data trajectory
%cl.v = cluster value for each data value
N = max(cl.b);
% cl.w = [];
% cl.p = [];
% cl.v = [];
% for i = 1:N
%     [w, p, v]  = newWeight(mdp, opts);
%     cl.w(:, i) = w;
%     cl.p(:, i) = p;
%     cl.v(:, i) = v;
% end

for i = 1:N
    indicesFill = find(cl.b == i); % vector containing indices value equal to i
    [w, p, v]  = newWeight(mdp, opts);

    w_T = w';

    weight_hash = key_enc(w_T);
    % compute Q from V
    Q_vals = QfromV(v, mdp, w);

    BQ1 = opts.eta .* Q_vals;
    BQsum1 = log(sum(exp(BQ1), 2));
    NBQ1   = bsxfun(@minus, BQ1, BQsum1);

    Q_cache.put(weight_hash, NBQ1);

    for j = 1:length(indicesFill)
        cl.w(:, indicesFill(j)) = w;
        cl.p(:, indicesFill(j)) = p;
        cl.v(:, indicesFill(j)) = v;
    end
end

end
