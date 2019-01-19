% Test 3
%% 
% ANN Logic Gate - OR
% Author: Balasubramanyam Evani
%%

rng('default');
rng(1);

Xi = [0, 0, 0; 0, 0, 1; 0, 1, 0; 0, 1, 1; 1, 0, 0;1, 0 , 1; 1,1,0;1,1,1];
yi = [0;1;1;1;1;1;1;1];
bk = 0.5;

xi = zeros(size(Xi,1),size(Xi,2)+1);
xi(:,1) = bk*ones(size(xi,1),1);

for col=1:size(Xi,2)
    
    xi(:,col+1) = Xi(:,col);
    
end

w = randn(1,size(xi,2));
eta = 0.6;
epochs = 350;

for itr=1:epochs
    for i=1:size(xi , 1)
        input = xi(i,:);
        sum = dot(w, input);
        yk = sign(sum);
        error = yk -yi(i);
        w = w - eta*error*sign_grad(yk)*input;
    end
end

disp(w);

test = [bk,0,0,0];
y = sign(dot(w,test));
disp(y);

function [res] = sign(val)
    a = -1;
    res = (1+exp(-val)).^-1;
end

function [res] = sign_grad(val)
    res = val*(1-val);
end

%%