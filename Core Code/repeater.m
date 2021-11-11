function  [CV,mCV] = repeater(X,Y,fs,cross,train,predict,n)
%% Repetition for K-fold Cross Validation
%------------------------------------------------------------------------------------------------%
% - Z.K.X. 2018/09/04
%------------------------------------------------------------------------------------------------%
%% Input
%  (1) X: feature (subjects * variates double matrix) 
%  (2) Y: label (subjects * variates double matrix) 
%  (3) fs
%      -- fs.type: feature selection strategy
%                [1] 'all' - all significant features
%                [2] 'separate' - feature separation based on the correlation direction (default)
%                [3] 'mean' - feature integration (mean value) based on the correlation direction
%                [4] 'no' - no operation (i.e., include all inputted features)
%    -- fs.FS: parameter 'FS' in script 'FStool'
%    -- fs.force: parameter 'force' in script 'FStool'
%  (4) cross: how many folds you want to create for the k-fold cross validation
%         (letting this parameter to be empty means conducting a leave-one-out cross validation)
%  (5) train
%      -- train.type: parameter 'S_type' in script 'UMAT' or 'trainer'
%      -- train.parameter: parameter 'Parameter' in script 'UMAT' or 'trainer'
%--------------------------------------------------------------------------
%  Common Algorithms (train.type)
%  [1] Binary Classification
%      {1} 'SVM' - Support Vector Machine                      
%      {2} 'GLM' - Logistic Regression (require extra setting for parameter)
%  [2] Regression
%      {1} 'regress1s' - Basic Regression
%      {2} 'GLM' - General Linear Model
%      {3} 'glmnet'/ 'cvglmnet' - Elastic Network Regression
%      {4} 'KRR' - Kernel Ridge Regression
%      {5} 'RVR' - Revelance Vector Regression
%--------------------------------------------------------------------------
%  (6) predict
%      -- predict.parameter: parameter 'Parameter' in script 'predictor'
%      -- predict.force: parameter 'force' in script 'predictor'
%  (7) n: number of repetition (default value = 100)
%------------------------------------------------------------------------------------------------%
%%
if (nargin < 3) 
    fs = [];
end
if (nargin < 4) 
    cross = 10;
end
if (nargin < 5) 
    train = [];
end
if (nargin < 6) 
    predict = [];
end
if (nargin < 7) 
    n = 100;
end

%%
f = 0;
for i = 2:n+1
    f = f + 1/n;
    cv = CV2(X,Y,fs,cross,train,predict,[f,i-1]);
    if ~strcmp(fs.type,'no')
        mCV(i-1).Feature = cv.feature;
        mCV(i-1).FN = cv.FN;
    end
    mCV(i-1).PE = cv.PE;
    a = length(mCV(i-1).PE); 
    if ~strcmp(fs.type,'no')
        b = length(mCV(i-1).FN);
    end
    if a < 6 & a > 1 
        s = size(mCV(i-1).PE,1);
        mCV(i-1).PE = [mCV(i-1).PE,zeros(s,6-a)];
        mCV(i-1).FN = [mCV(i-1).FN,zeros(s,6-b)];
    end
end

[a,b] = size(mCV(1).PE);
CV.PE = zeros(a,b);

if ~strcmp(fs.type,'no')
    [a,b] = size(mCV(1).PE);
    CV.FN = zeros(a,b);
    for i = 1:a 
        for j = 1:b
            CV.feature.sum{i,j} = zeros(1,size(X,2));
            CV.feature.mean{i,j} = zeros(1,size(X,2));
        end
    end
end

for i = 1:length(mCV)
    CV.PE = CV.PE + mCV(i).PE;
    if ~strcmp(fs.type,'no')
        CV.FN = CV.FN + mCV(i).FN;
        [a,b] = size(cv.feature.sum);
        for j = 1:a 
            for k = 1:b
                try
                    CV.feature.sum{j,k} = CV.feature.sum{j,k} + mCV(i).Feature.sum{j,k};
                    CV.feature.mean{j,k} = CV.feature.mean{j,k} + mCV(i).Feature.mean{j,k};
                catch
                    CV.feature.sum{j,k} = CV.feature.sum{j,k};
                    CV.feature.mean{j,k} = CV.feature.mean{j,k};                    
                end
            end
        end
    end
end

CV.PE  = CV.PE /i;
if ~strcmp(fs.type,'no')
    CV.FN  = CV.FN /i;
end

CV.Input.fs = fs; CV.Input.cross = cross; CV.Input.train = train; 
CV.Input.predict = predict; CV.Input.n = n; 

if ~strcmp(fs.type,'no')
    for i = 1:a 
        for j = 1:b
            CV.feature.sum{i,j} = CV.feature.sum{i,j};
            CV.feature.mean{i,j} = CV.feature.mean{i,j}/length(mCV);
        end
    end
end    

PE = CV.PE;
PE(PE<0) = 0;

figure1 = figure;       
for i = 1:size(PE,1)
    subplot1 = subplot(size(PE,1),1,i);
    hold(subplot1,'on');
    bar(PE(i,:));
    if i == 1
        title('Predictive Effect (All Feature)');
    elseif i == 2
        title('Predictive Effect (Positive Feature)');
    elseif i == 3
        title('Predictive Effect (Negative Feature)');
    end
    box(subplot1,'on');
    set(subplot1,'XTick',[1 2 3 4 5 6],'XTickLabel',{'0.05','0.01','0.005','0.001','0.0005','0.0001'});
end
    