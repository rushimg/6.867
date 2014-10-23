function [X,Y] = bishopCurveData()
%y = sin(2 pi x) + N(0,0.3),
data = importdata('curvefitting.txt');

X = data(1,:);
Y = data(2,:);

figure;

plot(X, Y, 'o', 'MarkerSize', 10);
xlabel('x');
ylabel('y');