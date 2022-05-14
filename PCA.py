"""
Max Deng
05/10/2022
"""
from random import random
from turtle import clear
import numpy as np #numpy import
###utilizing sklearn for implemented PCA/sPCA algorithms###
from sklearn.decomposition import PCA 
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import csv #utilizing csv reader, may change for other file formats.

#A simple python package I/O based csv reader for raw input data (see contents of arbitrary.csv)
def simplefileReader(filename):
    data_arr = [] #initializing array for use in PCA algorithms
    with open(filename) as file: #opening file
        dlist = csv.reader(file) #reading file line by line
        #next(dlist) ; NEEDED IF THERE ARE NON DATA POINT HEADERS
        data_arr = ([[int(x) for x in line[:]]for line in dlist]) #reformating read data into algo func compatible int array
    return data_arr #return for use.



if __name__ == '__main__': #file main
    #data = simplefileReader("/Users/maxdeng/Documents/mxd_Gerstein/Sparsity_Control/arbitrary.csv")

    np.random.seed(0)
    data_dense = np.random.rand(100,100) #creating a randomized set of values to fill a dense dataset
    data_dense [:, 2 * np.arange(50)] = 0 #inputing values into the dense dataset
    data_sparse = csr_matrix(data_dense) #Fitting values into a Compressed Sparse Row matrix as a form of sparsity control
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42) #Initializing algorithm using Singlar Value Decomposition as a form of dimensionality reduction
    svd.fit(data_sparse) #fitting the data into the sklearn provided algorithm

    pca = PCA(n_components=5,  random_state=42) #Initializing sklearn PCA algorithm with similar parameters to ensure comparable results
    pca.fit(data_dense) #fitting data into PCA algorithm. 

    pcaEVR = pca.explained_variance_ratio_
    pcaSVAL = pca.singular_values_
    svdEVR = svd.explained_variance_ratio_
    svdSVAL = svd.singular_values_

    print("\nPCA Explained Variance Ratios:" + str(pcaEVR)  + "\n") #Explained Variance Ratios: A % of variance within individually selected values
    print("PCA Singular Values:" + str(pcaSVAL) + "\n") #The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.
    print("TruncatedSVD PCA Variance Ratios:" + str(svdEVR) + "\n") #Explained Variance Ratios: A % of variance within individually selected values
    print("TruncatedSVD PCA Singular Values:" + str (svdSVAL)+ "\n") #The singular values corresponding to each of the selected components. The singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.

"""
Concluding Comments:
Simple implementation of sklearn's PCA algorithms to perform a calculation of variance and individual selected values as a point of comparison. 
Utilization of a Singular Value Decomposition as a form of sparsity control and dimensionality reduction.
Simplified filereader implemented for raw CSV text (see arbitrary.csv)
ready for use in a Jupyter notebook style IDE that supports in line data visualization and graph generation with pandas packages
also ready for rudimentary python supported graphics generation for simplistic graph and data plotting. 
"""