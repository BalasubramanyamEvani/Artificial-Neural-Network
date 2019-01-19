%%  Test 1
%%
% ANN Linear Regression 2D on Car MPG Data

% Author: Balasubramanyam Evani

%%

filename = 'mpg.csv';
data = readtable(filename);
N = size(data);

feature = (table2array(data(:,5)));
mpg = (table2array(data(:,1)));

rows = N(1);
cols = 2;

input(:,1) = ones(rows,1);
input(:,2) = feature;

input(:,2) = (input(:,2) - min(input(:,2)))/ max(input(:,2));
epochs = 200;

scatter(input(:,2) , mpg , 'filled');
hold on;

w = zeros(1,cols);
eta = 0.001;

for i=1:epochs
    
    gradient = zeros(1,cols);
    
    for j=1:rows
 
        xi = input(j,:);
        yi = mpg(j);
        h = dot(xi , w)-yi;
        
        gradient = gradient + 2*xi*h;
 
    end
    
    w = w - eta*gradient;
    disp(w);
    x = linspace(min(input(:,2)) , max(input(:,2)) , 100);
    y = w(1) + w(2)*x;
    c = plot(x,y);
    drawnow;
    delete(c);
    
end

fprintf('\nweights, w0 = %f , w1 = %f\n', w(1),w(2));

%%