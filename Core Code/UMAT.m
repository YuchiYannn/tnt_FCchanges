function R = UMAT(X,Y,C,S_type,M_type,Parameter)
%% Univariate & Multivariate Analysis Toolkit
%------------------------------------------------------------------------------------------------%
% - Z.K.X. 2019/01/09
%------------------------------------------------------------------------------------------------%
%% Dependency
%-- Robust Correlation Toolbox 
%                              (https://sourceforge.net/projects/robustcorrtool/)
%-- Multiple Comparison Correction Toolbox (compiled by Z.K.X.)
%-- mattest_tool Script (writen by Z.K.X.)
%-- LIBSVM -- A Library for Support Vector Machines 
%                              (https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
%-- Codes from Lei Du (Department of Radiology and Imaging Sciences, Indiana University)
%   'Structured sparse canonical correlation analysis for brain imaging genetics: 
%    an improved GraphNet method'
%-- Glmnet for Matlab (2013) Qian, J., Hastie, T., Friedman, J., Tibshirani, R. and Simon, N.
%                              (http://www.stanford.edu/~hastie/glmnet_matlab/)
%-- Kernel Ridge Regression in Matlab version 1.0.0.0 by Joseph Santarcangelo
%                              (https://ww2.mathworks.cn/matlabcentral/fileexchange/49989-kernel-ridge-regression-in-matlab)
%-- Revelance Vector Regression 
%                              (https://github.com/ZaixuCui/Pattern_Regression)
%-- Sparse Canonical Correlation Analysis Based on Accelerated Linearized Bregman Iteration
%   Chu, D., Liao, L. Z., Ng, M. K., & Zhang, X. (2013). 
%   Sparse canonical correlation analysis: new formulation and algorithm. 
%   IEEE transactions on pattern analysis and machine intelligence, 35(12), 3050-3065.
%   (https://web.bii.a-star.edu.sg/~zhangxw/Publications.html)
%----------------------------------------------------------------------------------------
%% Input
%-- X: independent variable (subjects * variates matrix) or gourp 1 (subjects * variates matrix)
%-- Y: dependent variable (subjects * variates matrix) or gourp 2 (subjects * variates matrix)
%      [if comparison between 2 group, it is also allowed to put all data in X, while put the 
%       corresponding labels in Y (low - high)]
%-- C: control variable (subjects * 1)
%````````````````````````````````````````````````````````````````````````````````````````
%-- S_type: statistical method
%``````````````````````````Univariate Approaches`````````````````````````````````````````
%   (01)Pearson Correlation: 'pearson' (default) / 'pearson-par'
%   (02)Spearman Correlation: 'spearman'
%   (03)Partial Correlation: 'partial correlation'
%   (04)Percentage Bend Correlation: 'PBC'
%   (05)Independent-samples T Test: 'independent T'
%   (06)Pared-samples T Test: 'paired T'
%   (07)Regression: 'regress1'
%   (08)mattest: 'mattest'
%   (09)Linear Model: 'LM1'
%   (10)Canonical Correlation Analysis: 'CCA1'
%   (11)Skipped Correlation: 'SC'
%``````````````````````````Multivariate Approaches```````````````````````````````````````
%   (01)Regression: 'regress1s'
%   (02)Generalized Linear Model: 'GLM'
%   (03)Canonical Correlation Analysis: 'CCA1s'
%   (04)Structured Sparse Canonical Correlation Analysis with GraphNet method: 'gnSCCA'
%   (05)Sparse Canonical Correlation Analysis Based on Accelerated Linearized Bregman Iteration: 'CCA_ALBI'
%   (06)Solving CCA Through the Standard Eigenvalue Problem: 'CCA-SEP'
%   (07)Solving CCA Through the Generalised Eigenvalue Problem: 'CCA-GEP'
%   (08)Solving CCA Using the SVD: 'CCA-SVD'
%   (09)Solving regularised canonical correlation analysis through the generalised eigenvalue problem: 'rCCA-GEP'
%       (with parameter optimization)
%   (10)Solving regularised canonical correlation analysis through the standard eigenvalue problem: 'rCCA-SEP'
%       (with parameter optimization)
%   (11)Solving kernel canonical correlation analysis through the generalised eigenvalue problem: 'kCCA-GEP'
%       (with parameter optimization)
%   (12)Solving kernel canonical correlation analysis through the standard eigenvalue problem: 'kCCA-SEP'
%       (with parameter optimization)
%   (13)Principl Component Anlysis: 'PCA'
%``````````````````````````````Machine Learning `````````````````````````````````````````
%   (01)Support Vector Machine: 'SVM'
%   (02)Kernel Ridge Regression: 'KRR'
%   (03)Revelance Vector Regression: 'RVR'
%   (04)Generalized Linear Model via Penalized Maximum Likelihood: 'glmnet'
%   (05)Generalized Linear Model via Penalized Maximum Likelihood with K-fold Cross-validation: 'cvglmnet'
%   (06)Lasso or Elastic Net Regularization: 'lasso'
%   (07)Ridge Regression: 'ridge'
%````````````````````````````````````````````````````````````````````````````````````````
%-- M_type: multiple comparison correction approaches
%   (01)Bonferroni-Holm Correction
%      M_type{1,1} = 'Holm'; M_type{1,2} = desired alpha level
%   (02)False Discovery Rate
%      M_type{1,1} = 'FDR'; M_type{1,2} = desired alpha level
%   (03)Family Wise Error
%      M_type{1,1} = 'FWE'; M_type{1,2} = desired alpha level
%````````````````````````````````````````````````````````````````````````````````````````
%-- Parameter: parameters for different methods
%``````````````````````````Univariate Approaches`````````````````````````````````````````
%   (01)'regress1'
%      Parameter = k (k:degree of a polynomial; default = 1) 
%   (02)'LM1'
%      Parameter = expression of linear model (e.g., 'Y~X1+X1^2+X2+X1*X2')
%      [NOTE: Y must be the target variate (Y; X1 must be the most focused variate X);
%             X2...... are the covariants (C) arranging in order]
%   (03)'mattest'
%      Parameter{1} = VarTypeValue; Parameter{2} = PermuteValue; Parameter{3} = BootstrapValue;
%      Parameter{4} = ShowhistValue; Parameter{5} = ShowplotValue
%   (04)'independent T'/'paired T'
%      Parameter{1} = 'Name' in 'ttest' or 'ttest2'
%      Parameter{2} = 'Value' in 'ttest' or 'ttest2'
%``````````````````````````Multivariate Approaches```````````````````````````````````````
%   (01)'GLM'
%      Parameter{1} = 'Modelspec' or 'Terms Matrix' in the function of 'fitglm';
%      Parameter{2} = 'Name-Value Pair Arguments' in function of 'fitglm'(sequential input in cell);
%                       (https://ww2.mathworks.cn/help/stats/fitglm.html#namevaluepairarguments)
%      [Note: for logistic binomial regression, the 'Y' input must be 0 or 1 as category  labels]
%   (02)'gnSCCA' 
%       [Parameters, should be tuned before running (default setting as following).]
%       Parameter.alpha1 = 0.1;
%       Parameter.alpha2 = 0.1;
%       Parameter.lambda1 = 1;
%       Parameter.lambda2 = 1;
%       Parameter.beta1 = 1;
%       Parameter.beta2 = 1;
%   (03)'CCA_ALBI' 
%       Parameter.deltaL    -- A integer which denotes the difference (L - m). 
%                             The default value is 0.
%       Parameter.L         -- The number of columns in Wx and Wy.
%       Parameter.mu_x      -- Parameter mu_x in the Linearized Bregman method. 
%       Parameter.mu_y      -- Parameter mu_y in the Linearized Bregman method.
%                              The default value is mu = 5.
%       Parameter.delta     -- Parameter delta in the Linearized Bregman method. delta must satisfy 0 < delta <1.
%                              The default value is 0.9.
%       Parameter.epsilon1  -- Tolerance for computing SWx. 
%       Parameter.epsilon2  -- Tolerance for computing SWy. 
%                              The default values are 1e-6. 
%   (04)'rCCA-GEP'/ 'rCCA-SEP' 
%       Parameter.c1  -- testing range for regularisation parameter c1 (default=[0.01:0.01:1]).
%       Parameter.c2  -- testing range for regularisation parameter c2 (default=[0.01:0.01:1]).
%       Parameter.reps  -- number of repetitions for cross-validation(default=50).
%   (05)'kCCA-GEP'/ 'kCCA-SEP' 
%       Parameter.Gaussian  -- sets the kernel width for Gaussian gram.
%                              {1}'median' -- median trick 
%                              {2}'none' -- width selected by user 
%       Parameter.sigma  -- if 'none', Parameter.Gaussian has been set to 'none', set the width.
%       Parameter.rels -- number of relations of interest (default==3).
%       Parameter.fold -- number of folds (default==5).
%``````````````````````````````Machine Learning `````````````````````````````````````````
%  （01）'SVM'
%      Parameter = setting for the training model （e.g., '-s 3 -t 2'）
%      -s svm_type : set type of SVM (default 0)
%     	             0 -- C-SVC
%	                 1 -- nu-SVC
%	                 2 -- one-class SVM
%	                 3 -- epsilon-SVR
%	                 4 -- nu-SVR
%      -t kernel_type : set type of kernel function (default 2)
%	                 0 -- linear: u'*v
%	                 1 -- polynomial: (gamma*u'*v + coef0)^degree
%	                 2 -- radial basis function: exp(-gamma*|u-v|^2)
%	                 3 -- sigmoid: tanh(gamma*u'*v + coef0)
%      -d degree : set degree in kernel function (default 3)
%      -g gamma : set gamma in kernel function (default 1/num_features)
%      -r coef0 : set coef0 in kernel function (default 0)
%      -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%      -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%      -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
%      -m cachesize : set cache memory size in MB (default 100)
%      -e epsilon : set tolerance of termination criterion (default 0.001)
%      -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
%      -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%      -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
%   (02)'KRR'
%       Parameter.ker = type of kernel
%                       [1]inear - 'lin' (default)
%                       [2]polynomial - 'poly' 
%                       [3]Radial basis function - 'rbf' 
%                       [4]sam kenal - 'sam'
%       Parameter.parameter = power value
%                             [1]Linear kernels bias:b
%                                Parameters = [b 1];                      
%                                k(x,z) = (b + xTz)
%                             [2]Polynomial
%                                Parameters = [b d];
%                                k(x,z) = (b+xTz)d
%                             [3]Radial basis function
%                                Parameters = [sigma]
%       Parameter.RegulationTerm = this value is determine imperially using cross validation
%   (03)'glmnet'
%       Parameter.family = 'family' in the function of 'glmnet'
%                 [Response variable. Quantitative (column vector) for family =
%                 'gaussian' (default), or family = 'poisson'(non-negative counts). 
%                 For family = 'binomial' should be either a column vector with two 
%                 levels, or a two-column matrix of counts or proportions. For
%                 family = 'multinomial', can be a column vector of nc>=2
%                 levels, or a matrix with nc columns of counts or proportions.
%                 For family = 'cox', y should be a two-column matrix with the
%                 first column for time and the second for status. The latter
%                 is a binary variable, with 1 indicating death, and 0
%                 indicating right censored. For family = 'mgaussian', y is a
%                 matrix of quantitative responses.]
%       Parameter.options = 'options' in the function of 'glmnet'
%                 [A structure that may be set and altered by glmnetSet 
%                 (default setting as following).]
%                 options.alpha = 1.0  (elastic-net mixing parameter)
%                 options.nlambda = 100  (number of lambda values)
%                 options.lambda depends on data, nlambda and
%                 lambda_min(user spplied lambda sequence) 
%                 options.standardize = true  (variable standardization)
%                 options.weights = all ones vector (observation weights)
%   (04)'cvglmnet'
%        Parameter.family = 'family' in the function of 'cvglmnet' and 'glmnet'
%        Parameter.options = 'options' in the function of 'cvglmnet' and 'glmnet'
%        Parameter.type = 'type'in the function of 'cvglmnet'
%                 [Loss to use for cross-validation. Currently five options, not all available 
%                 for all models. The default is type='deviance', which uses squared-error 
%                 for Gaussian models (a.k.a type='mse' there), deviance for
%                 logistic and Poisson regression, and partial-likelihood for 
%                 the Cox model. type='class' applies to binomial and multinomial 
%                 logistic regression only, and gives misclassification error.] 
%                 type='auc' is for two-class logistic regression only, and gives 
%                 area under the ROC curve. type='mse' or type='mae' (mean absolute error) 
%                 can be used by all models except the 'cox'; they measure the deviation 
%                 from the fitted mean to the response.]  
%        Parameter.nfolds = 'nfolds'in the function of 'cvglmnet'
%                 [Number of folds - default is 10. Although nfolds can be as large as the 
%                  sample size (leave-one-out CV), it is not recommended
%                  for large datasets. Smallest value allowable is nfolds=3.]
%        Parameter.foldid = 'foldid'in the function of 'cvglmnet'
%                  [An optional vector of values between 1 and nfold identifying
%                   what fold each observation is in. If supplied, nfold can be
%                   missing.]
%        Parameter.parallel = 'parallel'in the function of 'cvglmnet'
%                  [If true, use parallel computation to fit each fold. If a worker pool 
%                   is not open, it will open using the default cluster profile and close 
%                   after the computation is over.] 
%        Parameter.keep = 'keep'in the function of 'cvglmnet'
%                  [If keep=true, a prevalidated array is returned containing
%                   fitted values for each observation and each value of lambda.
%                   This means these fits are computed with this observation and
%                   the rest of its fold omitted. The foldid vector is also
%                   returned. Default is keep=false.]
%        Parameter.grouped = 'grouped'in the function of 'cvglmnet'
%                  [This is an experimental argument, with default true, and can
%                   be ignored by most users. For all models except the 'cox',
%                   this refers to computing nfolds separate statistics, and then
%                   using their mean and estimated standard error to describe the
%                   CV curve. If grouped=false, an error matrix is built up at
%                   the observation level from the predictions from the nfold
%                   fits, and then summarized (does not apply to type='auc'). 
%                   For the 'cox' family, grouped=true obtains the CV partial 
%                   likelihood for the Kth fold by subtraction; by subtracting the 
%                   log partial likelihood evaluated on the full dataset from that 
%                   evaluated on the on the (K-1)/K dataset. This makes more efficient 
%                   use of risk sets. With grouped=FALSE the log partial likelihood is 
%                   computed only on the Kth fold.]
%   (05)'lasso'
%        Parameter.alpha = weight of lasso versus ridge optimization [(0,1], default is 1]
%        for more detailes, see (https://ww2.mathworks.cn/help/stats/lasso.html)
%------------------------------------------------------------------------------------------------%
%% Output
%[1] General Outputs  
%-- r_value: r values for each feature 
%-- r2_value: r2 values for each feature 
%-- t_value: t values for each feature 
%-- F_value: F values for each feature 
%-- p_value: p values for each feature 
%-- p_value_corrected: corrected p values for each feature 
%-- sig_variate_corrected: serial numbers of features survived from multiple comparison correction
%-- Parameter: setting of 'Parameter'
%-- S_type: setting of 'S_type' 
%-- M_type: setting of 'M_type' 
%-- sig_results: information of significant results
%-- Model: output of model (coefficients) 
%-- mdl: output of model parameters
%[2] Canonical Correlation Outputs  
%   (01)'CCA_ALBI'
%       W_x   -- Nonsparse projection vectors for X. 
%       W_y   -- Nonsparse Projection vectors for Y. 
%       SW_x  -- Sparse projection vectors for X.
%       SW_y  -- Sparse projection vectors for X. 
%       corr  -- the list of correlation coefficients between training data in the projected spaces of X and Y. 
%   (02)'CCA-SEP'/ 'CCA-GEP'/ 'CCA-SVD'
%       za: the image of the position wa (a.k.a. canonical variate)
%       zb: the image of the position wb (a.k.a. canonical variate)
%       wa: the position in the data space of view a
%       wb: the position in the data space of view b
%       cc: the canonical correlation (cosine of the enclosing angle)
%       L: the Bartlett-Lawley statistic to be compared against the chi-squared distribution with (p-k)(q-k) degrees of freedom
%       prob: 1 - chi2cdf(L(i),(p-k(i))*(q-k(i))); [p- number of variables in view a; q- number of variables in view b]
%       criteria: critical values from the chi-squared distribution
%------------------------------------------------------------------------------------------------%
%% Automatic Inputting
if (nargin < 3)
    C = [];
end
if (nargin < 4)
    S_type = [];
end
if (nargin < 5)
    M_type = [];
end
if (nargin < 6)
    Parameter = [];
end
if isempty(S_type) == 1
    S_type = 'pearson-par';
end
if isempty(M_type) == 1
    M_type{1} = 'FDR';
end

%% Default Setting for Method
if strcmp(S_type,'pearson') | strcmp(S_type,'spearman') | strcmp(S_type,'partial correlation') |...
    strcmp(S_type,'PBC') | strcmp(S_type,'regress1') | strcmp(S_type,'LM1') |...
    strcmp(S_type,'CCA1') 
    execute = 1;
elseif strcmp(S_type,'mattest') 
    execute = 2;
elseif strcmp(S_type,'independent T') | strcmp(S_type,'paired T') 
    execute = 3;
elseif strcmp(S_type,'regress1s') | strcmp(S_type,'GLM') | strcmp(S_type,'CCA1s') | strcmp(S_type,'gnSCCA')...
        | strcmp(S_type,'glmnet') | strcmp(S_type,'cvglmnet') | strcmp(S_type,'CCA_ALBI')...
        | strcmp(S_type,'CCA-SEP') | strcmp(S_type,'CCA-GEP') | strcmp(S_type,'CCA-SVD') |...
        strcmp(S_type,'rCCA-GEP') | strcmp(S_type,'rCCA-SEP') | strcmp(S_type,'kCCA-SEP')|...
        strcmp(S_type,'kCCA-GEP') | strcmp(S_type,'PCA') | strcmp(S_type,'lasso') | strcmp(S_type,'ridge')
    execute = 4;
elseif strcmp(S_type,'SVM') | strcmp(S_type,'KRR') | strcmp(S_type,'RVR')
    execute = 5;
elseif strcmp(S_type,'pearson-par')
    execute = 11;
else
    execute = 6;
end

%% Default Setting for Parameter
if strcmp(S_type,'regress1') & isempty(Parameter) == 1
    Parameter = 1;
elseif strcmp(S_type,'LM1') & isempty(Parameter) == 1
    Parameter = 'Y~X1';
end
%
if strcmp(S_type,'GLM') & isempty(Parameter) == 1
    if isempty(C) == 1 
        Parameter{1} = 'linear';
    else
        T = zeros(size(X,2));
        for i = 1:size(X,2)
            T(i,i) = 1;
        end
        T = [T,zeros(length(T),size(C,2))];
        T = [T;zeros(size(C,2),size(T,2))];
        T = [zeros(1,length(T));T];
        Parameter{1} = T;
    end
    if isempty(C) ~= 1
        X = [X,C];
    end
    Parameter{2} = [];
end
%
if isempty(Parameter) == 1 & strcmp(S_type,'SVM')
    if (mean(rem(Y,1)) ~= 0) | (mean(rem(Y,1)) == 0 & mean(unique(Y)) > 6);
        Parameter = '-s 3';
    else
        Parameter = '-s 0';
    end
end
%
if isempty(Parameter) == 1 & strcmp(S_type,'lasso')
    Parameter.alpha = 1;
end 
%
if isempty(Parameter) == 1 & strcmp(S_type,'ridge')
    Parameter.lambda = linspace(0,1);
    Parameter.N = 5;
elseif ~isempty(Parameter) & strcmp(S_type,'ridge')
    a = fieldnames(Parameter);
    if isempty(find(strcmp(a,'lambda')))
        Parameter.lambda = linspace(0,1);
    end
    if isempty(find(strcmp(a,'N')))
        Parameter.N = 5;
    end    
end 
%
if isempty(Parameter) == 1 & strcmp(S_type,'KRR')
    Parameter.ker = 'lin';
    Parameter.parameter = [1 1];
    Parameter.RegulationTerm = 1;
elseif isempty(Parameter) ~=1 & strcmp(S_type,'KRR')
    a = fieldnames(Parameter);
    if isempty(find(strcmp(a,'ker')))
        Parameter.ker = 'lin';
    end   
    if isempty(find(strcmp(a,'parameter')))
        if strcmp(Parameter.ker,'lin')
            Parameter.Parameter = [1 1];
        elseif strcmp(Parameter.ker,'poly')
            Parameter.Parameter = [1 2];
        else
            Parameter.Parameter = [1 1];
        end
    end
    if isempty(find(strcmp(a,'Parameter.RegulationTerm')))
        Parameter.Parameter.RegulationTerm = 1;
    end     
end
%
if strcmp(S_type,'mattest') 
    if isempty(Parameter) == 1
        Parameter{1} = []; Parameter{2} = []; Parameter{3} = []; Parameter{4} = []; Parameter{5} = [];
    elseif length(Parameter) < 2
        Parameter{2} = []; Parameter{3} = []; Parameter{4} = []; Parameter{5} = [];
    elseif length(Parameter) < 3
        Parameter{3} = []; Parameter{4} = []; Parameter{5} = [];
    elseif length(Parameter) < 4
        Parameter{4} = []; Parameter{5} = [];
    elseif length(Parameter) < 5
        Parameter{5} = [];
    end
end
%
if strcmp(S_type,'independent T') | strcmp(S_type,'paired T') 
    if isempty(Parameter) 
        Parameter{1} = 'Tail'; Parameter{2} = 'both';
    elseif length(Parameter) < 2
        Parameter{2} = 'both';
    end
end
%
if strcmp(S_type,'rCCA-SEP') | strcmp(S_type,'rCCA-GEP')
    if isempty(Parameter)
        Parameter.c1 = [0.01:0.01:1];
        Parameter.c2 = [0.01:0.01:1];
        Parameter.reps = 50;
    end
elseif strcmp(S_type,'kCCA-SEP') | strcmp(S_type,'kCCA-GEP')
    if isempty(Parameter)
        Parameter.Gaussian = 'median';
        Parameter.rels = 3;
        Parameter.fold = 5;
    end 
end

%% Default Setting for Multiple Comparison Correction
if (nargin < 5) | isempty(M_type) == 1
    M_type{1} = 'FDR'; M_type{2} = 0.05;
end

%% Execution for Univariate Analysis        
if execute == 11
    if strcmp(S_type,'pearson-par')
        for i = 1:size(X,2)
                x = X(:,i); y = Y; c = C;
                f = find(isnan(x));
                x(f) = []; y(f) = []; 
                [r_value(1,i),p_value(1,i)] = corr(x,y);
        end
    end
end
         
if execute == 1
    for i = 1:size(X,2)
        x = X(:,i); y = Y; c = C;
        f = find(isnan(x));
        x(f) = []; y(f) = []; 
        if isempty(C) ~= 1
            c(f,:) = [];
        end
        if strcmp(S_type,'pearson')
            [r_value(1,i),p_value(1,i)] = corr(x,y);
        elseif strcmp(S_type,'spearman')
            [r_value(1,i),p_value(1,i)] = corr(x,y,'type','Spearman');
        elseif strcmp(S_type,'partial correlation')
            [r_value(1,i),p_value(1,i)] = partialcorr(x,y,c);
        elseif strcmp(S_type,'PBC')
            [r_value(1,i),t_value(1,i),p_value(1,i)] = bendcorr(x,y,0);
        elseif strcmp(S_type,'regress1')
            S = x;
            x = ones(size(x,1),1);
            for k = 1:Parameter
                x = [x,S.^k];
            end
            [b,bint,r,rint,stats] = regress(y,x);
            r2_value(1,i) = stats(1); F_value(1,i) = stats(2); p_value(1,i) = stats(3);
        elseif strcmp(S_type,'LM1')
            tbl(:,1) = table(y);
            tbl.Properties.VariableNames{'Var1'} = 'Y';
            tbl(:,2) = table(x);
            tbl.Properties.VariableNames{'Var2'} = 'X1';
            if isempty(c) ~= 1 
                for i = 1:size(c,2)
                    tbl(:,i+2) = table(c(:,i));
                    tbl.Properties.VariableNames{['Var',num2str(i+2)]} = ['X',num2str(i+1)];
                end
            end
            lm = fitlm(tbl,Parameter);
            S = anova(lm,'summary');
            p_value(1,i) = table2array(S(2,5));
            F_value(1,i) = table2array(S(2,4));
            r2_value(2,i) = lm.Rsquared.Adjusted;
            r2_value(1,i) = lm.Rsquared.Ordinary;
            clear tbl;
        elseif strcmp(S_type,'CCA1')
            [A,B,r_value(1,i),U,V,stats] = canoncorr(x,y);
            p_value(1,i) = stats.p;
        end
        disp(['Feature ',num2str(i),' has been finished!']);
    end
elseif execute == 2
    if strcmp(S_type,'mattest')
        result = mattest_tool(X',Y',Parameter{1},Parameter{2},Parameter{3},Parameter{4},Parameter{5});
        p_value = result.pValues'; t_value = result.tScores';
    end
elseif execute == 3
    type3 = unique(Y);
    if length(unique(Y)) ~= 2
        for i = 1:size(X,2)
            x = X(:,i); y = Y(:,i);
            f = find(isnan(x));
            x(f) = []; y(f) = []; 
            if strcmp(S_type,'independent T')
                [a,p_value(1,i),c,d] = ttest2(x,y,Parameter{1},Parameter{2});
                t_value(1,i) = d.tstat;
            elseif strcmp(S_type,'paired T')
                [a,p_value(1,i),c,d] = ttest(x,y,Parameter{1},Parameter{2});
                t_value(1,i) = d.tstat;
            end
        end
    else
        low = find(Y == type3(1));
        high = find(Y == type3(2));
        XX = X(low,:);
        YY = X(high,:);
        for i = 1:size(XX,2)
            x = XX(:,i); y = YY(:,i);
            f = find(isnan(x));
            x(f) = []; y(f) = []; 
            if strcmp(S_type,'independent T')
                [a,p_value(1,i),c,d] = ttest2(x,y,Parameter{1},Parameter{2});
                t_value(1,i) = d.tstat;
            elseif strcmp(S_type,'paired T')
                [a,p_value(1,i),c,d] = ttest(x,y,Parameter{1},Parameter{2});
                t_value(1,i) = d.tstat;
            end
        end
    end
end
    
if length(M_type) == 1
    M_type{2} = 0.05;
end

if exist('F_value') == 0
    F_value = [];
end
if exist('r_value') == 0
    r_value = [];
end
if exist('r2_value') == 0
    r2_value = [];
end
if exist('t_value') == 0
    t_value = [];
end

if execute == 1 | execute == 2 | execute == 3 | execute == 11  
if strcmp(M_type(1),'Holm')
    [R.p_value_corrected, R.sig_variate_corrected] = bonf_holm(p_value,M_type{2});
elseif strcmp(M_type(1),'FDR')
    [a,b,sig_variate_corrected] = makeFDR(p_value,M_type{2});
    R.p_value_corrected = [];
    R.sig_variate_corrected = zeros(1,length(p_value));
    R.sig_variate_corrected(sig_variate_corrected) = 1;
elseif strcmp(M_type(1),'FWE')
    R.p_value_corrected = p_value * length(p_value);
    R.sig_variate_corrected = zeros(1,length(p_value));
    f = find(R.p_value_corrected <= M_type{2});
    R.sig_variate_corrected(f) = 1;
end
R.r_value = r_value; R.t_value = t_value; R.p_value = p_value; R.r2_value = r2_value;
R.S_type = S_type; R.M_type = M_type; R.Parameter = Parameter; R.F_value = F_value;

F = find(R.p_value < 0.05);
for i = 1:length(F)
    R.sig_results(i).position = F(i);
    if isempty(R.r_value) ~= 1
        R.sig_results(i).r = R.r_value(F(i));
    end
    if isempty(R.r2_value) ~= 1
        R.sig_results(i).r2 = R.r2_value(F(i));
    end
    if isempty(R.t_value) ~= 1
        R.sig_results(i).t = R.t_value(F(i));
    end
    if isempty(R.F_value) ~= 1
        R.sig_results(i).F = R.F_value(F(i));
    end
    R.sig_results(i).p = R.p_value(F(i));
    R.sig_results(i).p_05 = 1;
    if R.sig_results(i).p < 0.01
        R.sig_results(i).p_01 = 1;
    else
        R.sig_results(i).p_01 = 0;
    end
    if R.sig_results(i).p < 0.005
        R.sig_results(i).p_005 = 1;
    else
        R.sig_results(i).p_005 = 0;
    end    
    if R.sig_results(i).p < 0.001
        R.sig_results(i).p_001 = 1;
    else
        R.sig_results(i).p_001 = 0;
    end
    if R.sig_results(i).p < 0.0005
        R.sig_results(i).p_0005 = 1;
    else
        R.sig_results(i).p_0005 = 0;
    end
    if R.sig_results(i).p < 0.0001
        R.sig_results(i).p_0001 = 1;
    else
        R.sig_results(i).p_0001 = 0;
    end
end

if mean(R.sig_variate_corrected) ~= 0 & ~isempty(F)
    for i = 1:length(R.sig_results)
        if R.sig_variate_corrected(R.sig_results(i).position) == 1
            R.sig_results(i).survive_mc = 1;
        else
            R.sig_results(i).survive_mc = 0;
        end
    end
else
    if ~isempty(F)
        for i = 1:length(R.sig_results)
            R.sig_results(i).survive_mc = 0;
        end
    end
end
end
    
%
if execute == 6
    if strcmp(S_type,'SC')
        [R.r_value,R.t_value,R.mdl.h,R.mdl.outid,R.mdl.hboot,R.mdl.CI] = skipped_correlation(X,Y,0,0.05);
        F = find(R.mdl.h == 1); 
        for i = 1:length(F)
            R.sig_results(i).position = F(i);
            R.sig_results(i).r = R.r_value(F(i));
        end
    end
end

%% Execution for Multivariate Analysis    
if execute == 4
    for i = 1:size(X,1)
        f = find(isnan(X(i,:)));
        if isempty(f) == 1
            F(i) = 0;
        else
            F(i) = 1;
        end
    end
    f = find(F == 1);
    X(f,:) = [];
    Y(f,:) = [];
    if strcmp(S_type,'lasso')
        [B,FitInfo] = lasso(X,Y,'Alpha',Parameter.alpha,'CV',10);
        R.idxLambda1SE = FitInfo.Index1SE;
        R.coef = B(:,R.idxLambda1SE);
        R.coef0 = FitInfo.Intercept(R.idxLambda1SE);
    elseif strcmp(S_type,'ridge')
        if length(Parameter.lambda) > 1
            [best_lambda,best_error] = cvrr(X,Y,Parameter.lambda,Parameter.N);
            R.model = ridge_regression(X,Y,best_lambda); 
            R.lambda = best_lambda; R.error = best_error;
        else
            R.model = ridge_regression(X,Y,Parameter.lambda);
            R.lambda = Parameter.lambda;
        end
    elseif strcmp(S_type,'regress1s')
        X = [ones(size(X,1),1),X];
        [B,BINT,R1,RINT,STATS] = regress(Y,X);
        R.Model = B';
        R.r2_value = STATS(1);
        R.F_value =  STATS(2);
        R.p_value = STATS(3);
    elseif strcmp(S_type,'PCA')
        V = corrcoef(X);
        [COEFF,latent,explained] = pcacov(V);
        R.eigenvalue(1,:) = {'特征值', '差值', '贡献率', '累积贡献率'};
        R.eigenvalue(2:size(X,2)+1,1) = num2cell(latent);
        R.eigenvalue(2:size(X,2),2) = num2cell(-diff(latent));
        R.eigenvalue(2:size(X,2)+1,3:4) = num2cell([explained, cumsum(explained)]);
        R.loading = COEFF(:,1:size(X,2));
        R.contribution = explained;
        for i = 1:length(find(latent>1))
            R.RT(:,i) = mean((X.*R.loading(:,i)'),2);
%             R.RT(:,i) = sum((X.*R.loading(:,i)'),2);
        end
    elseif strcmp(S_type,'GLM')
        P1 = 'R.mdl =  fitglm(X,Y,Parameter{1}';
        P2 = '';
        if isempty(Parameter{2}) ~= 1
            for i = 1:length(Parameter{2})
                eval([Parameter{2}{i} '=' 'Parameter{2}{i}']);
                P2 = [P2 ',' Parameter{2}{i}];
            end
        end
        P = [P1,P2,')'];
        eval(P);
        [R.p_value,R.F_value] = coefTest(R.mdl);
        R.r2_value(1) = R.mdl.Rsquared.Adjusted;
        R.r2_value(2) = R.mdl.Rsquared.Ordinary;
        R.Parameter = Parameter;
        R.Model = table2array(R.mdl.Coefficients(:,1))';
    elseif strcmp(S_type,'CCA1s')
        [Coefficient_X,Coefficient_Y,r,U,V,stats] = canoncorr(X,Y)
        R.r_value = r; R.p_value = stats.p;
        R.mdl.Coefficient_X = Coefficient_X; R.mdl.Coefficient_Y = Coefficient_Y; 
        R.mdl.U = U; R.mdl.V = V; R.mdl.stats = stats; 
        R.mdl.Canonical_loadings_X = corr(X,U);
        R.mdl.Canonical_loadings_Y = corr(Y,V);
        R.mdl.Cross_loadings_X = corr(X,V);
        R.mdl.Cross_loadings_Y = corr(Y,U);
        subplot(321);
        stem(Coefficient_X);
        title('Weights of X');
        subplot(322);
        stem(Coefficient_Y);
        title('Weights of Y');
        subplot(323);
        stem(R.mdl.Canonical_loadings_X);
        title('Canonical Loadings of X');
        subplot(324);
        stem(R.mdl.Canonical_loadings_Y);
        title('Canonical Loadings of Y');
        subplot(325);
        stem(R.mdl.Cross_loadings_X);
        title('Cross Loadings of X');
        subplot(326);
        stem(R.mdl.Cross_loadings_Y);
        title('Cross Loadings of Y');        
    elseif strcmp(S_type,'gnSCCA')
        X = getNormalization(X);
        Y = getNormalization(Y);
        if isempty(Parameter) == 1
            paras.alpha1 = 0.1;
            paras.alpha2 = 0.1;
            paras.lambda1 = 1;
            paras.lambda2 = 1;
            paras.beta1 = 1;
            paras.beta2 = 1;
        else
            paras = Parameter;
        end
        [u, v, R.mdl.error, R.mdl.U, R.mdl.V] = agn_scca(X, Y, paras);
        for i = 1:length(u)
            X2(:,i) = X(:,i) * u(i);
        end
        for i = 1:length(v)
            Y2(:,i) = Y(:,i) * v(i);
        end
        X2 = sum(X2,2); Y2 = sum(Y2,2); [R.r_value,R.p_value] = corr(X2,Y2);
        R.mdl.Coefficient_X = u; R.mdl.Coefficient_Y = v;
        R.mdl.Canonical_loadings_X = corr(X,X2);
        R.mdl.Canonical_loadings_Y = corr(Y,Y2);
        R.mdl.Cross_loadings_X = corr(X,Y2);
        R.mdl.Cross_loadings_Y = corr(Y,X2);
        subplot(321);
        stem(u);
        title('Weights of X');
        subplot(322);
        stem(v);
        title('Weights of Y');
        subplot(323);
        stem(R.mdl.Canonical_loadings_X);
        title('Canonical Loadings of X');
        subplot(324);
        stem(R.mdl.Canonical_loadings_Y);
        title('Canonical Loadings of Y');
        subplot(325);
        stem(R.mdl.Cross_loadings_X);
        title('Cross Loadings of X');
        subplot(326);
        stem(R.mdl.Cross_loadings_Y);
        title('Cross Loadings of Y');
    elseif strcmp(S_type,'CCA_ALBI')
        [R.Wx, R.Wy, R.SWx, R.SWy, R.corr] = CCA_ALBI(X',Y',Parameter); 
    elseif strcmp(S_type,'CCA-SEP')
        X_a = zscore(X); X_b = zscore(Y);
        [R.za,R.zb,R.wa,R.wb,R.cc] = cca_standard_eigenvalue_problem(X_a,X_b);
    elseif strcmp(S_type,'CCA-GEP')
        X_a = zscore(X); X_b = zscore(Y);
        [R.za,R.zb,R.wa,R.wb,R.cc] = cca_generalised_eigenvalue_problem(X_a,X_b);
    elseif strcmp(S_type,'CCA-SVD')        
        X_a = zscore(X); X_b = zscore(Y);        
        [R.za,R.zb,R.wa,R.wb,R.cc] = cca_svd(X_a,X_b);
    elseif strcmp(S_type,'rCCA-SEP') 
        [c1_opt,c2_opt,final] = repeated_cross_validation(X,Y,Parameter.c1,Parameter.c2,Parameter.reps); 
        X_a = zscore(X); X_b = zscore(Y); 
        [R.za,R.zb,R.wa,R.wb,R.cc] = cca_standard_regularised(X_a,X_b,c1_opt,c2_opt);
        R.c1_opt = c1_opt; R.c2_opt = c2_opt;
    elseif strcmp(S_type,'rCCA-GEP')
        [c1_opt,c2_opt,final] = repeated_cross_validation(X,Y,Parameter.c1,Parameter.c2,Parameter.reps); 
        X_a = zscore(X); X_b = zscore(Y); 
        [R.za,R.zb,R.wa,R.wb,R.cc] = cca_generalised_regularised(X_a,X_b,c1_opt,c2_opt); 
        R.c1_opt = c1_opt; R.c2_opt = c2_opt;
    elseif strcmp(S_type,'kCCA-GEP') | strcmp(S_type,'kCCA-SEP')
        X_a = zscore(X); X_b = zscore(Y); 
       [Ka,siga] = gaussK(X_a,Parameter.Gaussian); 
       [Kb,sigb] = gaussK(X_b,Parameter.Gaussian); 
       Ka = centering(Ka); Kb = centering(Kb);
       [mean_corr,c1_opt,c2_opt] = crossv_kernel(X_a,X_b,Parameter.fold); 
       if strcmp(S_type,'kCCA-GEP')
           [R.za,R.zb,R.wa,R.wb,R.cc] = cca_generalised_kernel(Ka,Kb,c1_opt,c2_opt,Parameter.rels);
       else
           [R.za,R.zb,R.wa,R.wb,R.cc] = cca_standard_kernel(Ka,Kb,c1_opt,c2_opt,Parameter.rels);
       end
       R.c1_opt = c1_opt; R.c2_opt = c2_opt;
    elseif strcmp(S_type,'glmnet') | strcmp(S_type,'cvglmnet')
        if isempty(Parameter) == 1
            Parameter.family = []; Parameter.options = []; Parameter.type = [];
            Parameter.nfolds = []; Parameter.foldid = []; Parameter.parallel = [];
            Parameter.keep = []; Parameter.grouped = [];
        else
            K = length(fieldnames(Parameter));
            if K < 2
                Parameter.options = []; Parameter.type = []; Parameter.nfolds = []; 
                Parameter.foldid = []; Parameter.parallel = [];
                Parameter.keep = []; Parameter.grouped = [];
            elseif K < 3
                Parameter.type = []; Parameter.nfolds = []; 
                Parameter.foldid = []; Parameter.parallel = [];
                Parameter.keep = []; Parameter.grouped = [];
            elseif K <4
                Parameter.nfolds = []; 
                Parameter.foldid = []; Parameter.parallel = [];
                Parameter.keep = []; Parameter.grouped = [];
             elseif K < 5
                Parameter.foldid = []; Parameter.parallel = [];
                Parameter.keep = []; Parameter.grouped = [];
            elseif K < 6
                Parameter.parallel = [];
                Parameter.keep = []; Parameter.grouped = [];
            elseif K < 7
                Parameter.keep = []; Parameter.grouped = [];
            elseif K < 8
                Parameter.grouped = [];
            end
        end
        if strcmp(S_type,'glmnet')
            R.mdl = glmnet(X,Y,Parameter.family,Parameter.options);
            f = find(R.mdl.dev == max(R.mdl.dev));
            R.r2_value = max(R.mdl.dev);
            k = glmnetCoef(R.mdl);
            R.Model = k(:,f)';
            R.nonzero_value = R.mdl.df(f);
        else
            R.mdl = cvglmnet(X,Y,Parameter.family,Parameter.options,Parameter.type,...
                Parameter.nfolds,Parameter.foldid,Parameter.parallel,Parameter.keep,Parameter.grouped);
            R.Model = cvglmnetCoef(R.mdl);
            a = ones(size(X,1),1);
            x = [a,X];
            x = x.*R.Model';
            x = sum(x,2);
            r = corr(x,Y);
            R.r2 = r * r;
            R.nonzero_value = length(find(R.Model~=0));
            R.Model = R.Model';
        end
    end
    if strcmp(S_type,'CCA-SEP') | strcmp(S_type,'CCA-GEP') | strcmp(S_type,'CCA-SVD') |...
            strcmp(S_type,'rCCA-SEP') | strcmp(S_type,'rCCA-GEP') | strcmp(S_type,'kCCA-SEP') |...
            strcmp(S_type,'kCCA-GEP') 
        n = size(X_a,1); p = size(X_a,2); q = size(X_b,2); 
        r = R.cc; k = 0:1:min(p,q)-1;
        for i = 1:size(k,2)
            L(i) = bartlett_lawley_statistic(n,k(i),p,q,r); 
            prob(i) = 1 - chi2cdf(L(i),(p-k(i))*(q-k(i)));
            criteria(i) = chi2inv(0.99,(p-k(i))*(q-k(i))); 
        end
        R.L = L; R.criteria = criteria; R.prob = prob;
    end
end

%% Execution for Machine Learning
if execute == 5
    for i = 1:size(X,1)
        f = find(isnan(X(i,:)));
        if isempty(f) == 1
            F(i) = 0;
        else
            F(i) = 1;
        end
    end
    f = find(F == 1);
    X(f,:) = [];
    Y(f,:) = [];
    if strcmp(S_type,'SVM')
        R.mdl = svmtrain(Y,X,Parameter);       
    elseif strcmp(S_type,'KRR')
        R.mdl = KernelRidgeRegression(Parameter.ker,X ,Parameter.parameter,Y,Parameter.RegulationTerm);
    elseif strcmp(S_type,'RVR')
        [R.Model,R.mdl] = W_Calculate_RVR(X, Y', C, 'None');
        R.Model = [0,R.Model];
    end
    R.Parameter = Parameter;    
end

%%
R.S_type = S_type;
