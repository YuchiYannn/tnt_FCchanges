function PT = permutator(X,Y,CV,n)
%% Permutation Test
%------------------------------------------------------------------------------------------------%
% - Z.K.X. 2018/08/21
%------------------------------------------------------------------------------------------------%
%% Input
% X: feature (subjects * variates double matrix) 
% Y: label (subjects * variates double matrix) 
% CV: output of 'CV2'
% n: number of permutation
%------------------------------------------------------------------------------------------------%
%% Output
% p_value: p value of permutation test
% PE_distribution: predictive effect value distribution
%------------------------------------------------------------------------------------------------%
%%
if (nargin < 4) | isempty(n)
    n = 1000;
end

%%
for j = 1:size(CV.PE,1)
    for k = 1:size(CV.PE,2)
        prediction_r{j,k} = zeros(n+1,1);
        prediction_r{j,k}(1,1) = CV.PE(j,k);
    end
end
            
f = 0;

for i = 2:n+1
    new_Y = Y(randperm(size(X,1)));
    f = f + 1/n;
    try
        if length(fieldnames(CV)) == 2 | length(fieldnames(CV)) == 4
            curr = repeater(X,new_Y,CV.Input.fs,CV.Input.cross,CV.Input.train,CV.Input.predict,CV.Input.n);
        else
            curr = CV2(X,new_Y,CV.Input.fs,length(CV.fold),CV.Input.train,CV.Input.predict,[f,i-1]);
        end
    catch
        curr.a = []; close all;
    end
    a = fieldnames(curr);
    if ~isempty(find(strcmp(a,'PE')))
        for j = 1:size(curr.PE,1)
            for k = 1:size(curr.PE,2)
                prediction_r{j,k}(i,1) = curr.PE(j,k);
            end
        end
    else
        for j = 1:size(prediction_r,1)
            for k = 1:size(prediction_r,2)
                prediction_r{j,k}(i,1) = -1;
            end
        end          
    end
    disp(['Permutation ',num2str(i-1),' has been finished!']);
end

for j = 1:size(CV.PE,1)
    for k = 1:size(CV.PE,2)
        sorted_prediction_r = sort(prediction_r{j,k}(:,1),'descend');
        position_sig = find(sorted_prediction_r == CV.PE(j,k));
        Permutation_p_pred_y_y = position_sig(1)/n;
        p_value(j,k) = Permutation_p_pred_y_y;
        r_value_distribution{j,k} = sorted_prediction_r;
    end
end    

PT.p_value = p_value; PT.PE_distribution = r_value_distribution;

%%
p = PT.p_value; pd = PT.PE_distribution;

l = [1:size(p,1) * size(p,2)];
l = reshape(l,size(p,1),size(p,2));

figure1 = figure;       
for i = 1:max(max(l))       
    subplot1 = subplot(size(p,1),size(p,2),i);
    hold(subplot1,'on');
    a = hist(pd{l==i},20);
    histfit(pd{l==i},20);
    line([CV.PE(l==i),CV.PE(l==i)],[0,max(a)],'LineWidth',3,'LineStyle','--','Color',[1 0 0]);
end