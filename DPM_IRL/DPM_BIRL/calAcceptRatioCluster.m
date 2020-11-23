function ratio = calAcceptRatioCluster(trajCnt, NBQ1, NBQ2)

% QL1    = QfromV(value1, mdp, w1);
% BQ1    = opts.eta.*QL1;
% BQsum1 = log(sum(exp(BQ1), 2));
% NBQ1   = bsxfun(@minus, BQ1, BQsum1);

% QL2    = QfromV(value2, mdp, w2);
% BQ2    = opts.eta.*QL2;
% BQsum2 = log(sum(exp(BQ2), 2));
% NBQ2   = bsxfun(@minus, BQ2, BQsum2);

x = 0;
if iscell(trajCnt)
    for m = 1:length(trajCnt)
        for t = 1:size(trajCnt{m}, 1)
            s = trajCnt{m}(t, 1);
            a = trajCnt{m}(t, 2);
            n = trajCnt{m}(t, 3);
            x = x + n*(NBQ1(s, a) - NBQ2(s, a));
        end
    end
% when only updating the cluster assignment
else
    for t = 1:size(trajCnt, 1)
        s = trajCnt(t, 1);
        a = trajCnt(t, 2);
        n = trajCnt(t, 3);
        x = x + n*(NBQ1(s, a) - NBQ2(s, a));
    end
end
% log() function for matlab returns the natural logarithm ln
% x = ln(pr(O_x | R_1) / pr(O_x) | R_2))
% exp(x) = pr(O_x|R_1) / pr(O_x) | R_2)
ratio = exp(x);

