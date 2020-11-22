import pandas
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import sklearn


def read_file(filename):
    with open(filename) as f:
        mat = []
        for line in f:
            vals = line.split(',')
            vec = []
            for val in vals:
                vec.append(int(val))
            mat.append(vec)
    return mat

def mat_cut(mat,low_x, high_x, low_y, high_y):
    ret = []
    for i in range(low_y, high_y):
        vec = []
        for j in range(low_x, high_x):
             vec.append(mat[j][i])
        ret.append(vec)

    return ret

def main():
    mat = read_file('new.txt')
    #gnd_trth_mat = mat_cut(mat, 23, 183, 27, 91)
    contact = 1
    plt.imshow(mat)
    plt.title('Predicted contact map')
    #plt.ylim(233, 297)
    #plt.xlim(23, 182)
    plt.show()



    print("done")

if __name__ == "__main__":
    main()



