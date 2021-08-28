import numpy as np

class LinearRegressionModel:
    def __init__(self, dims, learning_rate=0.05, n_estimators=100, batch_size=-1):
        self.dims = dims
        self.params = self.initial_parameters(self.dims)
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.batch_size = batch_size
    
    def initial_parameters(self, dims):
        np.random.seed(42)
        if type(dims) == int:
            dims = [dims]
        dims = dims + [1]
        params = {}
        for l in range(1, len(dims)):
            params['W'] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2/dims[l-1])
            params['b'] = np.zeros((dims[l], 1))
        return params
    
    def forward_propagation(self, X):
        params = self.params
        y_hat = np.dot(X, params['W'].T) + params['b']
        return y_hat
    
    def backward_propagation(self, X, y, y_hat):
        grads = {}
        m = X.shape[0]
        
        grads['dW'] = 2/m * np.dot((y_hat - y).T, X)
        grads['db'] = 2/m * np.sum(y_hat - y)
        return grads
    
    def update_params(self, grads):
        learning_rate=self.learning_rate
        params = self.params
        params['W'] = params['W'] - learning_rate*grads['dW']
        params['b'] = params['b'] - learning_rate*grads['db']
        return params
    
    def compute_cost(self, y, y_hat):
        m = y.shape[0]
        cost = 1/m * np.sum((y_hat - y)**2)
        return cost
    
    def fit(self, X, y, batch_size=-1):
        """
        Fit Linear Regression using Gradient Descent
        If - Model Batches size = -1 then Run Batch Gradient Descent
           - Model Batches size = 1  then Run Stochastic Gradient Descent
           - Model Batches size is some other numbers then run Mini-batch Gradient Descent
        Argument:
          -- X: A (m by n) numpy Array (or Matrix somehow) contains the input (features).
          -- y: A (m by 1) numpy Array contains the output (labels).
          -- m is the number of sample in the Trainning set, n is the number of features.
        """
        if (self.batch_size == -1):
            print('Fitting on Full Batch...')
            self.batch_fit(X, y)
        elif (self.batch_size == 1):
            print('Fitting using Stochastic Gradient Descent...')
            self.stochastic_fit(X, y)
        else:
            print('Fitting using Mini-Batch...')
            self.mini_batch_fit(X, y)
    
    def batch_fit(self, X, y):
        for i in range(self.n_estimators):
            y_hat = self.forward_propagation(X)
            curr_cost = self.compute_cost(y, y_hat)
            grads = self.backward_propagation(X, y, y_hat)
            
            self.params = self.update_params(grads)
            
            print('After {} estimators, Cost: {}'.format(i, curr_cost))
    
    def stochastic_fit(self, X, y):
        """
        Fit Linear Regression using Stochastic Gradient Descent (SGD)
        Argument:
          -- X: A (m by n) numpy Array (or Matrix somehow) contains the input (features).
          -- y: A (m by 1) numpy Array contains the output (labels).
          -- m is the number of sample in the Trainning set, n is the number of features.
        """
        for i in range(self.n_estimators):
            for j in range(X.shape[0]):
                X_curr, y_curr = X[j].reshape(1, X[j].shape[0]), y[j]
                y_hat = self.forward_propagation(X_curr)
                curr_cost = self.compute_cost(y_curr, y_hat)
                grads = self.backward_propagation(X_curr, y_curr, y_hat)
                self.params = self.update_params(grads)
                
                print('Epoch {}, After {} estimators, Cost: {}'.format(i, j, curr_cost))
                
    def mini_batch_fit(self, X, y):
        batches = self.mini_batch_split(X.shape[0], self.batch_size)
        for i in range(self.n_estimators):
            for j in range(len(batches)-1):
                batch_start = int(batches[j])
                batch_end = int(batches[j+1]) - 1

                X_curr, y_curr = X[batch_start:batch_end], y[batch_start: batch_end]
                y_hat = self.forward_propagation(X_curr)
                curr_cost = self.compute_cost(y_curr, y_hat)
                grads = self.backward_propagation(X_curr, y_curr, y_hat)
                self.params = self.update_params(grads)
                
                print('Epoch {}, After {} estimators, Cost: {}'.format(i, j, curr_cost))
                
                
    def mini_batch_split(self, m, batch_size=16):
        """
        Helper function, return the range of each mini-batch divided from batch size m
        """
        batches = np.linspace(0, (m // batch_size + 1) * batch_size, m // batch_size + 2)
        batches[-1] = m
        return batches
        
            
    def predict(self, X):
        return X * self.params['W'] + self.params['b']