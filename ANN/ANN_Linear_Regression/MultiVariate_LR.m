%%  Test 2
%%
% ANN Linear Regression on Car MPG Data

% Author: Balasubramanyam Evani

%%

filename = 'mpg.csv';
data = table2array(readtable(filename));
N = size(data);
rows = N(1);
cols = N(2);
mpg = data(:,1);

input = zeros(rows , cols);
input(:,1) = ones(rows,1);

for i=2:cols
    input(:,i) = (data(:,i) - min(data(:,i)))/max(data(:,i));  
end

w = zeros(1,cols);
epochs = 300;
eta = 0.001;

for itr=1:epochs
    
    gradient = zeros(1,cols);
    
    for j=1:rows
    
        xi = input(j,:);
        yi = mpg(j);
        h = dot(xi , w) - yi;
        gradient = gradient + 2*xi*h;
        
    end
    disp(w);
    w = w - eta*gradient;
    
end

sprintf('\n corresponding weight matrix: \n');
disp(w);
