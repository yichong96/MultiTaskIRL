function Q = QfromV(V, mdp, w)

nS = mdp.nStates;
nA = mdp.nActions;

if nargin < 3 || isempty(w)
    w = mdp.weight;
else
    mdp = convertW2RSepsis(w, mdp);
end

%if mdp.useSparse
%   Q = sparse(nS, nA);
%    for a = 1:nA
%        Q(:, a) = mdp.rewardS{a} + mdp.discount*(mdp.transitionS{a}(:, :)'*V);
%    end
%else

% Q(s,a) = r(s,a) + gamma * sum over all s'[T(s,a,s') * V(s')]
Q = zeros(nS, nA);
% size(squeeze(mdp.transition_reshape(:, :, 1))')
% size(V)
for a = 1:nA
   Q(:, a) = mdp.reward(:, a) ...
       + mdp.discount*(squeeze(mdp.transition_reshape(:, :, a))'* V);
   end

% Q = zeros(nS,nA);
% for s = 1:nS
%     for a = 1:nA
%     %V(s) = sum(squeeze(mdp.transition(s,piL(s), :)) .* (squeeze(mdp.reward(s,piL(s),:)) +  mdp.discount * V));
%     Q(s,a) = squeeze(mdp.transition(s,a,:))' * (squeeze(mdp.reward(s,a,:)) +  mdp.discount * V);
%     end
% end

end