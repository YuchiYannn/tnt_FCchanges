function P = predictor(X,Y,R,Parameter,force)
%% Predictive Analytics
% 
% - Z.K.X. 2018/08/18
%------------------------------------------------------------------------------------------------%
%% Input
%  X: test feature 
%  Y: test label
%  R: output of 'UMAT'
%  Parameter: parameters applied in some of the prediction model
%   (01)'glmnet'
%       Parameter.s = 's' in the function of 'glmnetPredict'
%                      [Value(s) of the penalty parameter lambda at which predictions
%                       are required. Default is the entire sequence used to create the model.]
%       Parameter.type = 'type' in the function of 'glmnetPredict'
%                      [Type of prediction required. Type 'link' gives the linear
%                       predictors for 'binomial', 'multinomial', 'poisson' or 'cox'
%                       models; for 'gaussian' models it gives the fitted values.
%                       Type 'response' gives the fitted probabilities for 'binomial'
%                       or 'multinomial', fitted mean for 'poisson' and the fitted
%                       relative-risk for 'cox'; for 'gaussian' type 'response' is
%                       equivalent to type 'link'. Type 'coefficients' computes the
%                       coefficients at the requested values for s. Note that for
%                       'binomial' models, results are returned only for the class
%                       corresponding to the second level of the factor response.
%                       Type 'class' applies only to 'binomial' or 'multinomial'
%                       models, and produces the class label corresponding to the
%                       maximum probability. Type 'nonzero' returns a matrix of
%                       logical values with each column for each value of s, 
%                       indicating if the corresponding coefficient is nonzero or not.]
%       Parameter.exact = 'exact' in the function of 'glmnetPredict'
%                      [If exact=true, and predictions are to made at values of s not
%                       included in the original fit, these values of s are merged
%                       with object.lambda, and the model is refit before predictions
%                       are made. If exact=false (default), then the predict function
%                       uses linear interpolation to make predictions for values of s
%                       that do not coincide with those used in the fitting
%                       algorithm. Note that exact=true is fragile when used inside a
%                       nested sequence of function calls. glmnetPredict() needs to
%                       update the model, and expects the data used to create it in
%                       the parent environment.]
%       Parameter.offset = 'offset' in the function of 'glmnetPredict'
%                      [If an offset is used in the fit, then one must be supplied
%                       for making predictions (except for type='coefficients' or
%                       type='nonzero')]
%   (02)'cvglmnet'
%       Parameter.s = 's' in the function of 'cvglmnetPredict'
%                      [Value(s) of the penalty parameter lambda at which predictions
%                       are required. Default is the value s='lambda_1se' stored on
%                       the CV object. Alternatively s='lambda_min' can be used. If s
%                       is numeric, it is taken as the value(s) of lambda to be used.]
%       Parameter.varargin = 'varargin' in the function of 'cvglmnetPredict'
%   (03)'SVM'
%       (https://blog.csdn.net/zhuikong/article/details/29805543)
%------------------------------------------------------------------------------------------------%
%% Output
% pv: predicted value
% tv: true value
% R: output in 'UMAT' (model information)
%------------------------------------------------------------------------------------------------%
%%
if isempty(Y)
    Y = sum(X,2);
end
if (nargin < 4)
    Parameter = [];
end
if (nargin < 5)
    force = [];
end

if strcmp(R.S_type,'cvglmnet')   
    if isempty(Parameter)
        Parameter.s = []; Parameter.type = []; Parameter.exact = []; Parameter.offset = []; Parameter.varargin = [];
    else
        a = fieldnames(Parameter);
        if isempty(find(strcmp(a,'s')))
            Parameter.s = [];
        end
        if isempty(find(strcmp(a,'type')))
            Parameter.type = [];
        end        
        if isempty(find(strcmp(a,'exact')))
            Parameter.exact = [];
        end   
        if isempty(find(strcmp(a,'offset')))
            Parameter.offset = [];
        end   
        if isempty(find(strcmp(a,'varargin')))
            Parameter.varargin = [];
        end
    end
end

%%
if strcmp(force,'Model')
    type = 'regress';
else
    if strcmp(R.S_type,'regress1s') | strcmp(R.S_type,'GLM') | strcmp(R.S_type,'glmnet') | strcmp(R.S_type,'RVR')        
        type = 'regress';
    elseif strcmp(R.S_type,'test') 
        P.pv = Y;
        type = 'nan';
    else
        type = 'machine';
    end
end

if strcmp(type,'regress') 
    curr = [ones(size(X,1),1),X];
    curr = curr.*R.Model;
    P.pv = sum(curr,2);
elseif strcmp(type,'machine')
    if strcmp(R.S_type,'cvglmnet')
        P.pv = cvglmnetPredict(R.mdl,X,Parameter.s,Parameter.type,Parameter.exact,Parameter.offset);
    elseif strcmp(R.S_type,'lasso')
        P.pv = X*R.coef + R.coef0;
    elseif strcmp(R.S_type,'ridge')
        P.pv = X*R.model;
    elseif strcmp(R.S_type,'SVM')
        [P.pv,P.svm_output,zz] = svmpredict(Y,X,R.mdl,Parameter);
        if length(unique(P.pv)) > 2
            P.r2 = P.svm_output(3); P.MSE = P.svm_output(2);
        else
            P.accuracy = P.svm_output(1);
        end
    elseif strcmp(R.S_type,'KRR')
        P.pv = KernelPrediction(R.mdl,X);
    elseif strcmp(R.S_type(1:3),'qa_')
        k = R.S_type(4:end);
        eval(['P.pv = test' k '(R.mdl,X);']);
    end    
end

P.R = R; P.tv = Y;