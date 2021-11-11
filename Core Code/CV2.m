function  CV = CV2(X,Y,fs,cross,train,predict,fig,index)
%% Cross Validation
%------------------------------------------------------------------------------------------------%
%  Original Version: Z.K.X. 2018/08/21
%  Latest Version: Z.K.X. 2019/06/28
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
%      {6} 'lasso' - Lasso or Elastic Net Regularization Regression
%--------------------------------------------------------------------------
%  (6) predict
%      -- predict.parameter: parameter 'Parameter' in script 'predictor'
%      -- predict.force: parameter 'force' in script 'predictor'
%  (7) fig: just ignore this setting
%  (8) index: grouping label for cross-valisation
%------------------------------------------------------------------------------------------------%
%% Output
%------------------------------------------------------------------------------------------------%
%%
% fs
if (nargin < 3) | isempty(fs)
    fs.type = 'separate';   % 'all'; 'separate'; 'mean'; 'no'
    fs.FS = [];
    fs.force = [];
end
a = fieldnames(fs);
if isempty(find(strcmp(a,'type')))
    fs.type = 'separate';
end
if isempty(find(strcmp(a,'FS')))
    fs.FS = [];
end
if isempty(find(strcmp(a,'force')))
    fs.force = [];
end
% cross
if (nargin < 4) | isempty(cross)
    cross = size(X,1);
end
% train
if (nargin < 5) | isempty(train)
    if length(unique(Y)) > 2     %C = unique(A) 返回与 A 中相同的数据，但是不包含重复项。
        train.type = 'regress1s';
    else
        train.type = 'SVM';
    end
    train.parameter = [];
end
a = fieldnames(train);
if isempty(find(strcmp(a,'type')))
    train.type = 'regress1s';
end
if isempty(find(strcmp(a,'parameter')))
    train.parameter = [];
end
% predict
if (nargin < 6) | isempty(predict)
    predict.parameter = [];
    predict.force = [];
end
a = fieldnames(predict);
if isempty(find(strcmp(a,'parameter')))
    predict.parameter = '[]';
end
if isempty(find(strcmp(a,'force')))
    predict.force = [];
end
% fig
if (nargin < 7) | isempty(fig)
    fig = [];
end
% index
if (nargin <8) | isempty(index)
    index = crossvalind('Kfold',size(X,1),cross); 
end

%%
fs_curr = zeros(cross,size(X,2));

if ~isempty(fig)
    H = multiwaitbar(2,[0 0],{'Please wait','Please wait'});   
else
    H = multiwaitbar(1,[0],{'Please wait'});
end
bar11 = 0;

for i = 1:cross
    Xtest = X(index == i,:);
    Ytest = Y(index == i,:);
    Xtrain = X(index ~= i,:);
    Ytrain = Y(index ~= i,:);
    if ~strcmp(fs.type,'no')
        FR = FStool(Xtrain,Ytrain,[],fs.FS,fs.force);
        if ~isempty(FR.fs)
            if strcmp(fs.type,'all')
                feature = FR.fs(1,:);
            elseif strcmp(fs.type,'separate')
                feature = FR.fs;
            elseif strcmp(fs.type,'mean')
                feature = FR.fs_sum;
            end
        else
            feature = [];
        end
    else
        feature{1} = Xtrain;
    end
    if ~isempty(feature)
    for j = 1:size(feature,1)
        for k = 1:size(feature,2)
            if ~isempty(feature{j,k})
                if ~strcmp(fs.type,'no')
                    feature_selection{j,k}(i,:) = zeros(1,size(X,2));
                    feature_selection{j,k}(i,FR.selected_feature{j,k}) = 1;
                    testX = Xtest(:,FR.selected_feature{j,k});
                    if strcmp(fs.type,'mean')
                        if j == 1
                            if ~isempty(FR.selected_feature{2,k})
                                a = mean(Xtest(:,FR.selected_feature{2,k}),2);
                            else
                                a = [];
                            end
                            if ~isempty(FR.selected_feature{3,k})
                                b = mean(Xtest(:,FR.selected_feature{3,k}),2);
                            else
                                a = [];
                            end                        
                            testX = [a,b];
                        else
                            testX = mean(Xtest(:,FR.selected_feature{j,k}),2);
                        end          
                    end
                else
                    testX = Xtest;
                end
                R{j,k} = trainer(feature{j,k},Ytrain,[],train.type,[],train.parameter);
                P{j,k} = predictor(testX,Ytest,R{j,k},predict.parameter,predict.force);
                predicted_value{j,k}{i} = P{j,k}.pv;
                E{j,k} = evaluator(Ytest,P{j,k},0);
                if length(P{j,k}.pv) > 1
                    cc = 1;
                    if length(unique(Y)) > 2 
                        r_value_set{j,k}(i) = E{j,k}.r_value; p_value_set{j,k}(i) = E{j,k}.p_value;
                    else                    
                        accuracy_set{j,k}(i) = E{j,k}.accuracy;
                        error_rate_set{j,k}(i) = E{j,k}.error_rate;
                        sensitive_set{j,k}(i) = E{j,k}.sensitive;
                        specificity_set{j,k}(i) = E{j,k}.specificity;
                        precision_set{j,k}(i) = E{j,k}.precision;
                        recall_set{j,k}(i) = E{j,k}.recall;
                    end
                else
                    cc = 2;
                end
            else
                R{j,k} = [];
                P{j,k} = [];
                E{j,k} = [];
            end
        end
    end
    end
    if ~strcmp(fs.type,'no')
        CV.fold(i).F = FR;
    end
    if ~isempty(feature)
        CV.fold(i).R = R; CV.fold(i).P = P; CV.fold(i).E = E;
    else
        CV.fold(i).R = []; CV.fold(i).P = []; CV.fold(i).E = [];
    end
    true_value{i} = Ytest;
    bar11 = bar11 + 1/cross;
    if ~isempty(fig)
        multiwaitbar(2,[fig(1) bar11],{['Operation ',num2str(fig(2))],['Fold ',num2str(i),' is on the process!']},H);
    else
        multiwaitbar(1,[bar11],{['Fold ',num2str(i),' is on the process!']},H);
    end
end
delete(H.figure);
clear('H');

CV.true_value = [];
for i = 1:cross
    CV.true_value = [CV.true_value;true_value{i}];
end

if exist('predicted_value')
for j = 1:size(predicted_value,1)
    for k = 1:size(predicted_value,2)
        if ~strcmp(fs.type,'no')
            CV.feature.sum{j,k} = sum(feature_selection{j,k});
            CV.feature.mean{j,k} = mean(feature_selection{j,k});
        end
        CV.predicted_value{j,k} = [];
        for i = 1:length(predicted_value{j,k})
            CV.predicted_value{j,k} = [CV.predicted_value{j,k};predicted_value{j,k}{i}];              
        end
        p.pv = CV.predicted_value{j,k};
        if length(p.pv) ~= length(CV.true_value)
            CV.evaluation{j,k} = [];
        else
            CV.evaluation{j,k} = evaluator(CV.true_value,p,0);
        end
        if cc ~= 2
            if length(unique(Y)) > 2
                CV.set.r_value{j,k} = mean(r_value_set{j,k});
                CV.set.p_value_set{j,k} = mean(p_value_set{j,k});
            else
                CV.set.accuracy{j,k}(i) = mean(accuracy_set{j,k});
                CV.set.error_rate = mean(error_rate_set{j,k});
                CV.set.sensitive = mean(sensitive_set{j,k});
                CV.set.specificity = mean(specificity_set{j,k});
                CV.set.precision = mean(precision_set{j,k});
                CV.set.recall = mean(recall_set{j,k});
           end
        end
    end
end

for j = 1:size(CV.evaluation,1)
    for k = 1:size(CV.evaluation,2)
        if ~isempty(CV.evaluation{j,k})
            if length(unique(Y)) > 2
                CV.PE(j,k) = CV.evaluation{j,k}.r_value;
            else
                CV.PE(j,k) = CV.evaluation{j,k}.accuracy;
            end
        end
        if ~strcmp(fs.type,'no')
            CV.FN(j,k) = length(find(CV.feature.sum{j,k}>0));
        end
    end
end
end

CV.Input.fs = fs; CV.Input.train = train; CV.Input.predict = predict; 

%%
if isempty(fig)

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

end