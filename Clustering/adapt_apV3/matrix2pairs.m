function M = matrix2pairs(Dist)
% Z.K.X. 2019/03/23
%%
nrow = length(Dist); 
nap = nrow*nrow-nrow;
M = zeros(nap,3);
j = 1;
for i=1:nrow
    for k = [1:i-1,i+1:nrow]
        M(j,1) = i;
        M(j,2) = k;
        M(j,3) = -Dist(i,k);
        j = j+1;
    end
end