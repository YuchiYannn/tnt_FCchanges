function E = evaluator(Y,P,fig)
%% Evaluation of Predictive Effect
% 
% - Z.K.X. 2018/08/18
%------------------------------------------------------------------------------------------------%
%% Input
% Y: true value of test label 
% P: output of 'predictor'
%------------------------------------------------------------------------------------------------%
%% Output
%  (1) Regression
%      -- r_value: correlation of true and predicted value
%      -- p_value: statistical significance of r value
%  (2) Binary Classification
%--------------------------------------------------------------------------
%                                 Predicted Category    Total                                 
%                                    1         -1
%
%                         1          TP        FN         P                   
%       True Category 
%                        -1          FP        TN         N
%
%                       Total        P'        N'        P+N
%--------------------------------------------------------------------------
%  Accuracy = ( TP + TN )/( P + N )
%  Error Rate = ( FP + FN )/( P + N ) = 1 - Accuracy
%  Sensitive = TP/P
%  Specificity = TN/N
%  Precision = TP/P'
%  Recall = TP/P = Sensive 
%  AUC: areas under the ROC curve
%  (3) Assesment
%   ConfusionMatrix: Confusion matrix of the classification process (True labels in columns, predictions in rows)
%   Kappa          : Estimated Cohen's Kappa coefficient
%   OA             : Overall Accuracy
%   varKappa       : Variance of the estimated Kappa coefficient
%   Z              : A basic Z-score for significance testing (considering that Kappa is normally distributed)
%   CI             : Confidence interval at 95% for the estimated Kappa coefficient
%------------------------------------------------------------------------------------------------%
%%
if (nargin < 3) | isempty(fig)
    fig = 1;
end

if length(unique(Y)) > 2
    type = 1;
else
    type = 2;
end

%%
if type == 1
    E.Assesment = assessment(Y, P.pv, 'regress');
    [E.r_value,E.p_value] = corr(P.pv,Y);
elseif type == 2
    E.Assesment = assessment(Y, P.pv, 'class');
    C = classindex(Y,P.pv);
    E.accuracy = C.A;
    E.error_rate = C.E;
    E.sensitive = C.S1;
    E.specificity = C.S2;
    E.precision = C.P;
    E.recall = C.R;
end

%%
if fig == 1
    if type == 1
        plot(Y,P.pv,'k.');
        xlabel('Observed Value');
        ylabel('Predicted Value');
        grid;
    elseif type == 2
        E.AUC = plot_roc(P.pv,Y);
    end
end