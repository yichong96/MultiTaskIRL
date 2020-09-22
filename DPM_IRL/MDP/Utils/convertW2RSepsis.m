function mdp = convertW2RSepsis(weight, mdp)

% weight refers to weight for all 6 features
%mdp.F is a [nS, nA, nS, f] matrix that stores the feature vectors of S,A,S transitions
%how to build a reward function from this feature matrix ?
mdp.weight = weight;

% quadrouple for loop
% ================================================================

% tic
% for s = 1:mdp.nStates
% 	for a = 1:mdp.nActions
% 		for nextS = 1: mdp.nStates
% 			for f = 1:mdp.nFeatures
% 				temp(s,a,nextS,f) = mdp.F(s,a,nextS,f) * weight(f);
% 			end
% 		end
% 	end
% end
% temp = sum(temp, 4);
% mdp.reward = temp;
% toc
% ================================================================


% 4D features for r(s,a,s')
% ================================================================

% temp_F = reshape(mdp.F, mdp.nStates*mdp.nActions*mdp.nStates, mdp.nFeatures);
% temp_R = temp_F * weight;	%(s*a*s',nF) * (nF, 1)
% mdp.reward = reshape(temp_R, mdp.nStates,mdp.nActions,mdp.nStates);
% ================================================================

% 3D features for r(s,a)
temp_F = reshape(mdp.F_reshape, mdp.nStates*mdp.nActions, mdp.nFeatures);
temp_R = temp_F * weight;
mdp.reward = reshape(temp_R, mdp.nStates, mdp.nActions);

end
