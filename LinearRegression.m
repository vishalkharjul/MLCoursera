clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
xdata = load('C:\Coursera Assignments\ex3Data\ex3x.dat');
ydata = load('C:\Coursera Assignments\ex3Data\ex3y.dat');
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


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

sigma = std(x);   %max-min
mu = mean(x);

%Subtracting column's mean from each column value and then deviding it by it's column's standard deviation
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%f %f %f], y = %f \n', [x(1:10,:) y(1:10,:)]');


fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 1.0;
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(x, y, theta, alpha, num_iters);

 %j_history will give cost
% Plot the convergence graph

figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

P = [1 1650 3];
Price = [1 (1650-mu(2))/sigma(2) (3-mu(3))/sigma(3)]*theta;


fprintf(['Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'], Price);

fprintf('Program paused. Press enter to continue.\n');
pause;


fprintf('Solving with normal equations...\n');

x = xdata;
y = ydata;
m = length(y);

% Add intercept term to X
x = [ones(m, 1) x];
theta = zeros(size(x, 2), 1);
theta = pinv(x'*x)*x'*y;

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
%price = 0; % You should change this
P = [1 1650 3];
Price = P * theta;



% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], Price);

