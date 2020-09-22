% Sepsis problem
function mdp = sepsis(discount)


nS = 750;                % # of states
nA = 25;                         % actions: north, south, west, east
nF = 6    % # of features. blocksize = 2 therefore nF = 16

load('./MDP/featuresSepsis');
load('./MDP/transitionSepsis');	%750x25x750
load('./MDP/reshaped_features'); % 750x25x6
load('./MDP/transitionSepsis_reshape');
mdp.name       = sprintf('Sepsis nStates: %d nActions: %d', nS, nA);
mdp.nStates    = nS;
mdp.nActions   = nA;
mdp.nFeatures  = nF;
mdp.discount   = discount;
mdp.transition = transMat;
mdp.transition_reshape = transMat_reshape;
mdp.F          = featureSepsis;
mdp.F_reshape = reshaped_features;
mdp.reward     = zeros(nS, nA) % R(s,a)

