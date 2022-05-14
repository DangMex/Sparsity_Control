from random import random
import numpy as np #numpy import
###utilizing sklearn for implemented PCA/sPCA algorithms###
from sklearn.decomposition import PCA 
from sklearn.decomposition import SparsePCA
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
    sdata = simplefileReader("/Users/maxdeng/Documents/mxd_Gerstein/Sparsity_Control/arbitrary.csv")
    pca = PCA(n_components=2)
    pca.fit(sdata)
    print("PCA explained variance ratios:")
    print(pca.explained_variance_ratio_)
    print("PCA singular values:")
    print(pca.singular_values_)