# Stochastic Gradient Descent for Linear Regression

This is a Python implementation of stochastic gradient descent (SGD) for linear regression. Compared to the previous batch gradient descent implementation, this SGD version has several advantages and disadvantages.

## Overview

The code performs the following steps:

1. Generates sample data with a linear relationship between `x` and `y`.
2. Initializes the model parameters `w` (slope) and `b` (intercept) to 0.0.
3. Defines a loop that runs the stochastic gradient descent algorithm for 1000 epochs.
   - In each epoch, the data is shuffled to prevent the algorithm from getting stuck in local minima.
   - For each sample, the gradients are computed, and the parameters are updated.
   - The loss is calculated and printed after each epoch.
4. Prints the final linear regression model.

## Advantages of Stochastic Gradient Descent

1. **Faster Convergence**: Stochastic gradient descent can converge faster, especially for large datasets, as it updates the parameters more frequently.
2. **Ability to Escape Local Minima**: The inherent noise in the stochastic gradients can help the algorithm escape local minima and find a better global minimum.
3. **Memory Efficiency**: Stochastic gradient descent only needs to load a single sample into memory at a time, making it more memory-efficient than batch gradient descent.

## Disadvantages of Stochastic Gradient Descent

1. **Noisier Updates**: The parameter updates in stochastic gradient descent are noisier than in batch gradient descent, which can lead to less stable convergence.
2. **Potential for Oscillation**: If the learning rate is too high, the parameter updates can oscillate around the minimum, leading to slower convergence or even divergence.
3. **More Hyperparameters**: Stochastic gradient descent introduces an additional hyperparameter, the number of epochs, which needs to be tuned along with the learning rate.

## Usage

To use the code, simply run the Python script. The code will generate some sample data, run the stochastic gradient descent algorithm, and print the final parameter values and the linear regression model.

```python
# Run the script
python stochastic-gradient.py
```

## The code can be customized by:

- Changing the size and distribution of the sample data (`x` and `y`)
- Adjusting the learning rate `learning_rate`
- Modifying the number of epochs `num_epochs`
- Experimenting with different batch sizes or mini-batch approaches
