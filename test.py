import numpy as np #numpy import
"""utilizing sklearn for implemented PCA/sPCA algorithms, will show understanding through comments"""
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import csv


def fileReader(filename):
    data_arr = [] #initializing array for use in PCA algorithms
    with open(filename) as file: #opening file
        dlist = csv.reader(file) #reading file line by line
        #next(dlist) ; NEEDED IF THERE ARE NON DATA POINT HEADERS
        data_arr = ([[int(x) for x in line[:]]for line in dlist]) #appending csv values to array for use by sklearn PCA algo
    return data_arr #return for use.

if __name__ == '__main__':
    darr = fileReader("/Users/maxdeng/Documents/mxd_Gerstein/Sparsity_Control/arbitrary.csv")
    dataset = np.array(darr)
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    print(dataset)
    print(data)