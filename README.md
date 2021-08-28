# simple-linear-regression-from-scratch
A simple Linear Regression I built to enhance my comprehension
---
# Update
 - **26/08/21**: Build a Simples Linear Regression with Batch Gradient Descent
 - **27/08/21**: 
     + Develop 3 Gradient Descent Tranining Techniques: Batch Gradient Descent, Stochastic Gradient Descent (SGD) and Mini-batch Gradient Descent.
     + Fix Previous Linear Model and build Polynomial Model on top of it.
 - **28/08/21**:
     + Rename files for easier tracking.
     + Apply ***He. Initialization***.
     + File Separation:
         + Separate `LinearRegressionModel` (wrote as a `Class`) into `t_model.py`.
         + Separate `polynomial_features()` function into `t_preprocessing.py`.
         + Separate `plot_polynomial_curve()` function into `t_visualization.py`.
     + Apply Ridge Regularization.
         + This Regularization worked, but not like what I was expected.
         + Instead of add the Derivative of Regularization term in to the grads 
             + `(grads['dW'] = 2/m * np.dot((y_hat - y).T, X) + self.lambd/m*self.params['W'])`
         + I subtracted it directly from the `params['W']` which look like this: 
             + `params['W'] = params['W'] - learning_rate*grads['dW'] - self.lambd/m*self.params['W']`
         + In the long run, it will be crashed guaranteedly. 
         + *Why I did this?*
             + I figured out that the learning rate of my model was too small $(\alpha = 2.10^{-11})$, multiply this baby number by the regularization terms seems helpless, just like cancel out the regularization term.
             + But if I set the $\alpha$ bigger, our model with dramaticly diverge.
         + I will find the way to fix this, or even find other solutions if it not improved.
             