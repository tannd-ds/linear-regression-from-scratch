import numpy as np
import matplotlib.pyplot as plt

def plot_polynomial_curve(model, color, label):
    """
    Plot a curve of Polynomial Model (W1*X + W2*X^2 + Wn*X^n + b)
    Parameters:
      -- model: An Object (make with t_model.LinearRegressionModel)
      -- color: Color of the plot.
      -- label: Name of the Plot, which served as a legend.
    """
    X_model = np.linspace(-2, 4, 200).reshape(200, 1)
    y_model = np.zeros(X_model.shape)
    for l in range(1, model.params['W'].shape[1]+1):
        y_model = y_model + model.params['W'][:, l-1] * X_model**l
    y_model += model.params['b']
    plt.plot(X_model, y_model, c=color, ls='-', label=label)