from random import random
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
    data_dense = np.random.rand(100,100)
    data_dense [:, 2 * np.arange(50)] = 0
    data_sparse = csr_matrix(data_dense)
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    svd.fit(data_sparse)

    pca = PCA(n_components=5, n_iter=7, random_state=42)
    pca.fit(data_dense)

    print("PCA explained variance ratios:")
    print(pca.explained_variance_ratio_)
    print("PCA singular values:")
    print(pca.singular_values_)