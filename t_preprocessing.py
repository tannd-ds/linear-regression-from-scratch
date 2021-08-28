import numpy as np

def polynomial_features(features, degree=2):
    """
    Return a numpy matrix contains normal features (input) and the polynomial features with nth to 2nd degree.
    """
    poly_features = features
    for curr_degree in range(2, degree+1):
        poly_features = np.concatenate((poly_features, features ** curr_degree), axis=1)
    
    return poly_features