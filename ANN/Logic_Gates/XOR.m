% Test 3
%% 
% ANN Logic Gate - XOR
% Author: Balasubramanyam Evani
%%

Xi = [0,0,0; 
      0,0,1; 
      0,1,0;
      0,1,1;
      1,0,0;
      1,0,1;
      1,1,0;
      1,1,1];
  
yi = [0;1;1;0;1;0;0;1];

xi = zeros(size(Xi,1), size(Xi,2)+1);

for i=1:size(Xi,2)
    xi(:,i) = Xi(:,i);
end
xi(:,size(xi,2)) = ones(size(xi,1),1);

input_size = size(xi,2);
hidden_size = 3;
output_size = 1;

eta = 0.5;

w1 = randn(input_size, hidden_size);
w2 = randn(hidden_size, output_size);

disp(size(w2));
epochs = 10000;

for itr=1:epochs

   l1 = xi * w1;
   act1 = sigmoid(l1);
   l2 = act1 * w2;
   y = sigmoid(l2);
   
   delta2 = (y-yi) .* (y.*(1-y));
   delta1 = (delta2 * w2.') .* (act1.*(1-act1));
   w2 = w2 - eta*(act1.'*delta2);
   w1 = w1 - eta*(xi.'*delta1);

end

z1 = xi * w1;
a1 = sigmoid(z1);
z2 = a1 * w2;
out = sigmoid(z2);
disp(out);

function [res] = sigmoid(val)
    a = 1;
    res = (1+exp(-a*val)).^-1;
end