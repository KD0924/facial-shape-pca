# -*- coding: utf-8 -*-
"""
Apply PCA to data in 2D array.
Each row assumed to be one sample
@author: Tim Cootes
"""

import numpy as np

def apply_pca(D,t=3):
    """Assume each row is a sample. Apply PCA to compute mean and modes.
    Return [mean,P,mode_var] to give model of form x=mean+P@b
    """
    
    # Calculate the mean shape by averaging over the first dimension
    mean_shape=D.mean(0)

    # First subtract mean from each row
    dD=D-mean_shape
    
    n_shapes=int(D.shape[0])

    # Create covariance matrix
    S=dD.T @ dD/n_shapes

    # Compute the eigenvectors and eigenvalues (arbitrary order)
    evals,EVecs = np.linalg.eigh(S)

    # Sort by the eigenvalues (largest first)
    idx = np.flip(np.argsort(evals),0)
    evals = evals[idx]
    EVecs = EVecs[:,idx]

    # Create a model with t modes
    P=EVecs[:,0:t]
    
    return [mean_shape,P,evals[0:t]]
