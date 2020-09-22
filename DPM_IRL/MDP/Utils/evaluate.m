function [V, H] = evaluate(piL, mdp, w)
%
% Evaluate policy measured on the given reward weight w
% V = (I-gamma*T^piL)^-1 R^piL
%
if nargin < 3 || isempty(w)
    w = mdp.weight;
end

nS = mdp.nStates;
nA = mdp.nActions;

%if mdp.useSparse
%    I = speye(nS);
%    Tpi = sparse(nS, nS);
%    for a = 1:nA
%        idx = find(piL == a);
%        if ~isempty(idx)
%            Tpi(idx, :) = mdp.transitionS{a}(:, idx)';
%        end
%    end
%    idx = (piL-1)*nS+(1:nS)';
%    idx = (idx-1)*nS+(1:nS)';
%    Epi = sparse(nS, nS*nA);
%    Epi(idx) = 1;
%else
%    I = eye(nS);
%    Tpi = zeros(nS, nS);
%    for a = 1:nA
%        idx = find(piL == a);
%        if ~isempty(idx)
%            Tpi(idx, :) = squeeze(mdp.transition(:, idx, a)');
%        end
%    end
%    idx = (piL-1)*nS+(1:nS)';
%    idx = (idx-1)*nS+(1:nS)';
%    Epi = zeros(nS, nS*nA);
%    Epi(idx) = 1;
%end

%if nS < 2^14
%    H = (I-mdp.discount.*Tpi)\(Epi*mdp.F);
%    (I-mdp.discount .* Tpi)H = (Epi * mdp.f)
%    size(H) = (nS x nF)
%

% Linear algebra policy evaluation
% ================================================
I = eye(nS);
Tpi = zeros(nS, nS);

% index rewards based on current policy
R = mdp.reward(piL);

% build transition matrix P(s|s')
for a = 1:nA
    idx = find(piL == a);
    if ~isempty(idx)
        Tpi(idx, :) = squeeze(mdp.transition_reshape(:, idx, a)');
    end
end

% computation of inverse
V = R \ (I - mdp.discount * Tpi);
V = V'; % change shape from (1x750) to (750x1)
H = 0;  % not sure what H is

% ================================================


%else
%    EPS       = 1e-12;
%    MAX_ITERS = 10^4;
%    EF = a *mdp.F;
%    H  = EF;
%    for iter = 1:MAX_ITERS
%        H_old = H;
%        H = EF + mdp.discount*(Tpi*H);
%        done = approxeq(H, H_old, EPS);
%        if done, break; end
%        if sum(isnan(H(:))) > 0, break; end
%    end
%end
%V = H*w;
%size(V) = nS x 1
%end

% Dynamic programming policy evaluation
% ================================================
% V = zeros(nS,1);
% H = 0;
% EPS = 1e-4;
% diff = -1;
% while 1
%     oldV = V;
%     for s = 1:size(V,1) % loop through all nStates
%         % sum over all s' T(s,a,s') * (r(s,a,s') + gamma * V(s'))
%         % disp(squeeze(mdp.transition(s,piL(s), :))
%         V(s) = sum(squeeze(mdp.transition(s,piL(s), :)) .* (squeeze(mdp.reward(s,piL(s),:)) +  mdp.discount * V));
%     end
%     diff = abs(V - oldV);
%     %disp(mean(diff))
%     if (diff < 0.005)
%         break
%     end
% ================================================
end % end while





