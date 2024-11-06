# Gradient Descent for Linear Regression

This is a Python implementation of gradient descent for linear regression. The goal is to find the best-fitting linear model that maps input features `x` to a target variable `y`.

## Overview

The code performs the following steps:

1. Generates sample data with a linear relationship between `x` and `y`.
2. Initializes the model parameters `w` (slope) and `b` (intercept) to 0.0.
3. Defines a `descend()` function that calculates the gradients of the loss function with respect to `w` and `b`, and updates the parameters accordingly.
4. Runs the gradient descent process for 800 iterations, updating the parameters at each step.
5. Prints the final values of the parameters and the input/target data.

## Key Concepts

1. **Linear Regression**: The goal is to find the best-fitting linear model `yhat = wx + b` that maps the input features `x` to the target variable `y`.

2. **Mean Squared Error Loss**: The loss function used is the mean squared error, which measures the average squared difference between the predicted values (`yhat`) and the true values (`y`).

3. **Gradient Descent**: The algorithm iteratively updates the model parameters `w` and `b` by taking steps in the direction of the negative gradient of the loss function. This helps minimize the loss and find the best-fitting linear model.

4. **Partial Derivatives**: The gradients `dldw` and `dldb` are calculated by taking the partial derivatives of the loss function with respect to the parameters `w` and `b`, respectively.

5. **Learning Rate**: The learning rate `0.01` determines the step size for each parameter update. A larger learning rate can lead to faster convergence, but it may also cause the algorithm to overshoot the minimum. A smaller learning rate can result in slower convergence but more stable updates.

## Usage

To use the code, simply run the Python script. The code will generate some sample data, run the gradient descent algorithm, and print the final parameter values and input/target data.

```python
# Run the script
python gradient_descent_linear_regression.py
```

## Customization

You can customize the code by:

- Changing the size and distribution of the sample data (`x` and `y`)
- Adjusting the learning rate `learning_rate`
- Modifying the number of iterations `for epoch in range(800)`
- Experimenting with different loss functions or optimization algorithms
