function hst1 = testSepsis(irlOpts)

% no random seed for problems

% hst1 = cell(length(problem.iters), 1);
% hst2 = cell(length(problem.iters), problem.newExps);

discount = 0.9
mdp  = sepsis(discount)

% only need matrix of size [no.trajs, trajLength, value/policy]
% load trajectory data
%data = generateDemonstration(mdp, problem);
load('./MDP/demonstrations_states_19.mat');
load('./MDP/demonstrations_actions_19.mat');
load('./MDP/transitionSepsis_reshape.mat');


%scale down problem
%demonstrations_states = demonstrations_states(1:2,:)
%demonstrations_actions = demonstrations_actions(1:2,:)
demonstrations_states = demonstrations_states_19(:, 1:20);	% last index is icustayid
demonstrations = cat(3,demonstrations_states,demonstrations_actions_19);
mdp.T = reshape(transMat_reshape, 750, 750*25);
[sol, ~, hst] = feval(irlOpts.alg, demonstrations, mdp, irlOpts); % feval -> matlab function




%   hst1{iter1}      = evalMEIRL(sol, data, mdp, problem);
%    hst1{iter1}.data = data;
%    hst1{iter1}.sol  = sol;

%    nTrajs = size(data.trajSet, 1);
%    vdiff = nan(nTrajs, 1);
%    for m = 1:nTrajs
%        kE = data.trajId(m);
%        wE = data.weight(:, kE);
%        pE = data.policy(:, kE);
%        [VE, HE] = evaluate(pE, mdp, wE);
%        vE = full(wE'*HE'*mdp.start);

%        kL = sol.belongTo(m);
%        wL = sol.weight(:, kL);
%        if isfield(sol, 'policy') && ~isempty(sol.policy)
%            pL = sol.policy(:, kL);
%        else
%            mdp = convertW2R(wL, mdp);
%            pL  = policyIteration(mdp);
%        end
%        [VL, HL] = evaluate(pL, mdp, wL);
%        vL = full(wE'*HL'*mdp.start);

%        vdiff(m) = vE - vL;
%    end
%    evd = mean(vdiff);
%    fprintf('## EVD: %12.4f\n\n', evd);

%     nTrajs = size(data.trajSet, 1);
%     valueDiff = zeros(size(hst.elapsedT));
%     for t = 1:length(hst.elapsedT)
%         cl = hst.cl{t};
%
%         vdiff = nan(nTrajs, 1);
%         for m = 1:nTrajs
%             kE = data.trajId(m);
%             wE = data.weight(:, kE);
%             pE = data.policy(:, kE);
%             [VE, HE] = evaluate(pE, mdp, wE);
%             vE = full(wE'*HE'*mdp.start);
%
%             kL = cl.b(m);
%             wL = cl.w(:, kL);
%             pL = cl.p(:, kL);
%             [VL, HL] = evaluate(pL, mdp, wL);
%             vL = full(wE'*HL'*mdp.start);
%
%             vdiff(m) = vE - vL;
%         end
%         valueDiff(t) = mean(vdiff);
%     end

%     % testing new trajectory
%     for iter2 = 1:problem.newExps
%         fprintf('++ %d ++\n', iter2);
%
%         % generate new data
%         problem.seed = problem.initSeed + iter2;
%         newdata = generateDemonstration(mdp, problem, data.weight);
%
%         % transfer clustered information
%         RandStream.setGlobalStream(RandStream.create('mrg32k3a', ...
%             'NumStreams', 1, 'Seed', problem.seed));
%         if ~isempty(strfind(irlOpts.alg, 'DPM_BIRL'))
%             wL = DPM_BIRL_transfer(hst, newdata.trajSet, mdp, irlOpts);
%         else
%             wL = feval(sprintf('%s_transfer', irlOpts.alg), ...
%                 sol, newdata.trajSet, mdp, irlOpts);
%         end
%
%         hst2{iter1, iter2}.data = newdata;
%         hst2{iter1, iter2}.wL   = wL;
%     end
end
