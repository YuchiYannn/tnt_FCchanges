%% Setup Wizard for Prediction Analysis via StAR-2
% 
%  Z.K.X. 2018/08/23 (compiled via MATLAB-2018a)
%--------------------------------------------------------------------------------------------------------------------%
%% --------------------------------------- Data Loading --------------------------------------- %%
clear;

X=
Y=

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
S.train.type ='SVM'
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
