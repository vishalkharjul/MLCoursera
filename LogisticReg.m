clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
xdata = load('C:\Coursera Assignments\LogRegression\ex4x.dat');
ydata = load('C:\Coursera Assignments\LogRegression\ex4y.dat');
x = xdata;
y = ydata;
m = length(y);   %total no of samples

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [x(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;


% Add intercept term to X
x = [ones(m, 1) x];

% find returns the indices of the
% rows meeting the specified condition
pos = find(y == 1); neg = find(y == 0);

% Assume the features are in the 2nd and 3rd
% columns of x
plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'o')
xlabel('Test1 Score');
xlabel('Test2 Score');

num_iters = 10;
% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Calculate the hypothesis function
    z = x * theta;
    h = sigmoid(z);

    
    % Calculate gradient and hessian.
    % The formulas below are equivalent to the summation formulas
    % given in the lecture videos.
    grad = (1/m).*x' * (h-y);
    H = (1/m).*x' * diag(h) * diag(1-h) * x;
    
    % Calculate J (for testing convergence)
    J_history(iter) =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h));
    
    theta = theta - H\grad;
end
% Display theta
theta

% calculate probability 
prob = 1 - sigmoid([1, 20, 80]*theta)

fprintf(['Probability of student with testscore1=20, testscore2=80 and not admitted ' ...
         '(using neuton method):\n %f\n'], prob);


% Plot Newton's method result
% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

% Plot J
figure
plot(0:num_iters-1, J_history, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J_history')
% Display J
J
