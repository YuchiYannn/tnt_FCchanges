%% Setup Wizard for Prediction Analysis via StAR-2
% 
%  Z.K.X. 2018/08/23 (compiled via MATLAB-2018a)
%--------------------------------------------------------------------------------------------------------------------%
%% --------------------------------------- Data Loading --------------------------------------- %%
clear;
load('data_house.mat');
X=x'
Y=y
%% --------------------------------------- Feature Preprocessing --------------------------------------- %%
%% set feature matrix (subjects x features)
M(1).X = X; 
%% set target variable (subjects x 1)
M(1).Y = Y; 
%% set covariate matrix (subjects x covariates)
M(1).C = F2; 
%% set parameter for data selection stept
%  [1] label: one column of cell array labels (cell/string) 
M(1).P.label = 'label';
%  [2] outlier: outlier detection setting
M(1).P.outlier.tool = 0;
%                             {1} 0 - no execute outlier detection (default)
%                             {2} 1 - execute outlier detection via Robust Correlation Toolbox 
%                                 (https://sourceforge.net/projects/robustcorrtool/)
%                             {3} 2 - execute outlier detection via MATLAB built-in function
M(1).P.outlier.method = [];
%                             /-------------     tool = 1     ----------/
%                             {1} 'boxplot' - relies on the interquartile range
%                             {2} 'MAD' - relies on the median of absolute distances
%                             {3} 'S-outlier' - relies on the median of absolute distances
%                             {4} 'All' - the 3 methods above will be computed (default)
%                             /-------------     tool = 2     ----------/
%                             {1} 'median'
%                             {2} 'mean' (default)
%                             {3} 'quartiles'
%                             {4} 'grubbs'            
%                             {5} 'gesd'
%                             (https://ww2.mathworks.cn/help/matlab/ref/isoutlier.html)
M(1).P.outlier.parameter = 1;
%                             /-------------     tool = 1     ----------/
%                             {1} 1 - do univariate outliner detection
%                             {2} 2 - do bivariate outliner detection 
%  [3] regress: if conduct regressing out covariates
M(1).P.regress = 1;
%                             {1} 0 - no execute regressing out covariates (default)
%                             {2} 1 - execute regressing out covariates
%  [4] normalize: data standardization setting
M(1).P.normalize.method = [];
%                             {1} [] - no execute normalization (default)
%                             {2} 'rescale' 
%                             (http://ww2.mathworks.cn/help/matlab/ref/rescale.html)
%                             {3} 'standard' / 'scaling' ('StatisticalNormaliz.m')
M(1).P.normalize.parameter = [0 1];
%                             {1} (http://ww2.mathworks.cn/help/matlab/ref/rescale.html)
%  [5] group: percentage of grouping boundaries
M(1).P.group = 0.27;
%% if conduct comparison of high and low groups
S.HL = 0;
%                             {1} 0 - NO  (regression issue)
%                             {2} 1 - YES (binary classification issue)
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%% Note:
%     The feature preprocessing stept allows multiple inputs. If there are
%     other features you want to add into analysis for a multimodal
%     prediction study, you can compile the inputs into subsequent M
%     structure. 
%
%     e.g., M(2).X = X2; M(2).Y = Y2; M(2).C = []; M(2).P
%     = []; M(3).X = X3; M(3).Y = Y3; M(3).C = []; M(3).P = []
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% ===============================================================================================
%% run feature preprocessing
if S.HL == 1
    D = DO(M);
    X = D.X(D.order ~= 0);
    Y = D.order(D.order ~= 0);
elseif S.HL == 0
    D = DO(M);
    X = D.X;
    Y = D.Y;
end
% ===============================================================================================
%% --------------------------------------- Feature Extraction --------------------------------------- %%
%% set parameter for feature fusion

%% set parameter for feature selection
%  [1] fs: if conduct feature selection
S.if_fs = 1;
%                             {1} 0 - NO  
%                             {2} 1 - YES 
%  [2] thr: set p value threshold of feature selection (include features with p value less than thr )
S.thr = 0.05;
%  [3] FS: feature selection strategy (allow multiple strategies:n * 3)
S.FS{1,1} = 'pearson';
%          common commands: 'pearson'/ 'spearman'/ 'independent T'/ 'regress1'/ 'LM1' (more details, see 'UMAT.mat')                       
S.FS{1,2} = [];
%          common commands: 'FDR'/'FWE' (more details, see 'UMAT.mat')  
S.FS{1,3} = [];
%                                       (more details, see 'UMAT.mat')  
S.FS{1,4} = 1;
%           FS{n,1} = S_type in UMAT 
%           FS{n,2} = M_type in UMAT ([]: feature selection without multiple comparison correction )
%           FS{n,3} = Parameter in UMAT
%           FS{1,4} = stragety of combination (1 - union set; 2 - intersection)
%% Note:
%     The feature selection stept allows multiple inputs. If there are
%     other strategies you want to add into analysis for enlarging the 
%     selected features (S.FS{1,4} = 1) or restricting the selected 
%     features (S.FS{1,4} = 2), you can compile the inputs into subsequent 
%     FS cell.    
%
%     e.g., if you want to add a 1uadratic feature selection strategy, then
%     you can compile the script in the following way
%
%     FS{2,1} = 'regress1';
%     FS{2,2} = [];
%     FS{2,3} = 2;            %  degree of polynomial         
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% ===============================================================================================
%% run feature extraction
if S.if_fs == 1
    FR = FStool(X,Y,[],S.FS);
    f = find(FR.UMAT_outputs.p_value < S.thr);
    X = X(:,f);
end
% ===============================================================================================
%% --------------------------------------- Cross Validation --------------------------------------- %%
%% set parameter for nested feature selection
%  [1] fs: strategy of nested feature selection
S.fs.type = 'separate';
%                 {1} 'all' - all significant features
%                 {2} 'separate' - feature separation based on the correlation direction (default)
%                 {3} 'mean' - feature integration (mean value) based on the correlation direction
%                 {4} 'no' - no operation (i.e., include all inputted features)
%  [2] FS: parameter 'FS' in script 'FStool'
S.fs.FS{1,1} = 'pearson';
%          common commands: 'pearson'/ 'spearman'/ 'independent T'/ 'regress1'/ 'LM1' (more details, see 'UMAT.mat')                       
S.fs.FS{1,2} =[];
%          common commands: 'FDR'/'FWE' (more details, see 'UMAT.mat')  
S.fs.FS{1,3} = [];
%                                       (more details, see 'UMAT.mat')  
S.fs.FS{1,4} = 1;
%           FS{n,1} = S_type in UMAT 
%           FS{n,2} = M_type in UMAT ([]: feature selection without multiple comparison correction )
%           FS{n,3} = Parameter in UMAT
%           FS{1,4} = stragety of combination (1 - union set; 2 - intersection)
%  [3] force: parameter 'force' in script 'FStool'
S.fs.force = [];
%% set parameter for cross validation
S.cross =10;            % S.cross = [];
%         'cross' means how many folds you want to create for the k-fold cross validation
%         (letting this parameter to be empty means conducting a leave-one-out cross validation)
%% set parameter for model training
%  [1] type: parameter 'S_type' in script 'UMAT' or 'trainer'
S.train.type ='RVR'
%  [2] parameter: parameter 'Parameter' in script 'UMAT' or 'trainer'
S.train.parameter = [];
%% set parameter for model prediction
%  [1] parameter: parameter 'Parameter' in script 'predictor'
S.predict.parameter = [];
%% Alternative Model (S.train.type):
%  Core Function
%  [1] Binary Classification
%      {1} 'SVM' - Support Vector Machine                      
%      {2} 'GLM' - Logistic Regression (require extra setting for parameter)
%  [2] Regression
%      {1} 'regress1s' - Basic Regression
%      {2} 'GLM' - General Linear Model
%      {3} 'glmnet'/ 'cvglmnet' - Elastic Network Regression
%      {4} 'KRR' - Kernel Ridge Regression
%      {5} 'RVR' - Revelance Vector Regression
%      {6} 'SVM' - Support Vector Regression    
%  Other Function
%  [1] Binary Classification
%  [2] Regression
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%% Note:
%  If you only want to test the whole CV framework except the algorithms,
%  you can make the setting as: S.train.type = 'test'. 
% ===============================================================================================
%% run cross validation 
    a = fieldnames(S);
    if ~isempty(find(strcmp(a,'train')))
        CV = CV2(X,Y,S.fs,S.cross,S.train,S.predict);
    end
% ===============================================================================================
%% --------------------------------------- Permutation Test --------------------------------------- %%
%% set permutation times
S.n = 1000;
% ===============================================================================================
%% run permutation test
a = fieldnames(S);
if ~isempty(find(strcmp(a,'n')))
    PT = permutator(X,Y,CV,S.n);
end
% ===============================================================================================
%% --------------------------------------- Model Generalization --------------------------------------- %%
%% set variate in group 1
X1 = X;
Y1 = Y;
%% set variate in group 2
X2 = X;
Y2 = Y;
% ===============================================================================================
%% run model generalization
R = trainer(X1,Y1,[],S.train.type,[],S.train.parameter);
P = predictor(X2,Y2,R,S.predict.parameter);
E = evaluator(Y2,P);
% ===============================================================================================