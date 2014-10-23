function regressionPlot(X,Y,order)
% X is an array of N data points (one dimensional for now), that is, Nx1
% Y is a Nx1 column vector of data values
% order is the order of the highest order polynomial in the basis functions
figure;

plot(X, Y, 'o', 'MarkerSize', 10);
xlabel('x');
ylabel('y');

% You will need to write the designMatrix and regressionFit functions

% constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
phi = designMatrix(X,order);
% compute the weight vector
w = regressionFit(X, Y, phi);

hold all

x = 0:.01:1;
y = w'*designMatrix(x,order)';
plot(x,y)