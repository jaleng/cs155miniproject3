# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import sys
import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta, u, ai, bj):
    return (1-reg*eta)*Ui + eta * Vj * ((Yij - u) - (np.dot(Ui,Vj) + ai + bj))     
def grad_V(Vj, Yij, Ui, reg, eta, u, ai, bj):
    return (1-reg*eta)*Vj + eta * Ui * ((Yij - u) - (np.dot(Ui,Vj) + ai + bj))
def grad_a(Yij, Ui, Vj, reg, eta, u, ai, bj, a): 
    return (1-reg*eta)*a + eta  * ((Yij - u) - (np.dot(Ui,Vj) + ai + bj))
def grad_b(Yij, Ui, Vj, reg, eta, u, ai, bj, b): 
    return (1-reg*eta)*b + eta  * ((Yij - u) - (np.dot(Ui,Vj) + ai + bj))


def get_err(U, V, Y, u, a, b, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V.
    """
    # Compute mean squared error on each data point in Y; include
    # regularization penalty in error calculations.
    # We first compute the total squared squared error
    err = 0.0
    for (i,j,Yij) in Y:
        err += 0.5 *((Yij - u) - (np.dot(U[i-1], V[:,j-1]) + a[i-1] + b[j-1]))**2
    # Add error penalty due to regularization if regularization
    # parameter is nonzero
    if reg != 0:
        U_frobenius_norm = np.linalg.norm(U, ord='fro')
        V_frobenius_norm = np.linalg.norm(V, ord='fro')
        a_frobenius_norm = np.linalg.norm(a)
        b_frobenius_norm = np.linalg.norm(b)


        err += 0.5 * reg * (U_frobenius_norm ** 2)
        err += 0.5 * reg * (V_frobenius_norm ** 2)
        err += 0.5 * reg * (a_frobenius_norm ** 2)
        err += 0.5 * reg * (b_frobenius_norm ** 2)
    # Return the mean of the regularized error
    return err / float(len(Y))

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=100):
    U = np.random.random((M,K)) - 0.5
    V = np.random.random((K,N)) - 0.5
    a = np.random.random((M,)) - 0.5
    b = np.random.random((N,)) - 0.5

    col3 = Y[:, 2]
    u = np.mean(col3)


    size = Y.shape[0]
    delta = None
    print("training reg = %s, eta = %s, k = %s, M = %s, N = %s"%(reg, eta, K, M, N))
    indices = range(size)    
    for epoch in range(max_epochs):
        # Run an epoch of SGD
        before_E_in = get_err(U, V, Y, u, a, b, reg)
        np.random.shuffle(indices)
        for ind in indices:
            (i,j, Yij) = Y[ind]
            # Update U[i], V[j]
            U[i-1] = grad_U(U[i-1], Yij, V[:,j-1], reg, eta, u, a[i-1], b[j-1])
            V[:,j-1] = grad_V(V[:,j-1], Yij, U[i-1], reg, eta, u, a[i-1], b[j-1])
            a = grad_a(Yij, U[i-1], V[:,j-1], reg, eta, u, a[i-1], b[j-1], a)
            b = grad_a(Yij, U[i-1], V[:,j-1], reg, eta, u, a[i-1], b[j-1], b)
            
        # At end of epoch, print E_in
        E_in = get_err(U, V, Y, u, a, b, reg)
        print("Epoch %s, E_in (MSE): %s"%(epoch + 1, E_in))

        # Compute change in E_in for first epoch
        if epoch == 0:
            delta = before_E_in - E_in
        elif epoch < 5:
            continue
        # If E_in doesn't decrease by some fraction <eps>
        # of the initial decrease in E_in, stop early            
        elif before_E_in - E_in < eps * delta:
            break

    return (U, V, get_err(U, V, Y, u, a, b))