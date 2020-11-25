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
                try:
                    vec.append(float(val))
                except:
                    pass
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

def binarize(matrix, threshold):
    for i in range(0, len(matrix)):
        line = matrix [i]
        for j in range(0, len(line)):
             if matrix[i][j]<=threshold:
                 matrix[i][j] = 0
             else:
                 matrix[i][j] = 1
    return matrix

def find_positive  (original_mat, predicted_mat):
    cor_org = []
    cor_pre = []

    for i in range(0, len(original_mat)):
        line = original_mat [i]
        for j in range(0, len(line)):
             if original_mat[i][j] ==1 :
                 cor_org.append((i,j))

    for i in range(0, len(predicted_mat)):
        line = predicted_mat[i]
        for j in range(0, len(line)):
             if predicted_mat[i][j] == 1:
                cor_pre.append((i, j))

    return cor_pre, cor_org

def true_positive(cor_pre, cor_org):
    true_p = []
    fal_p = []
    for i in range(0, len(cor_pre)):
        for j in range(0, len(cor_org)):
             if cor_pre [i] == cor_org [j] :
                 true_p.append((cor_pre [i]))

    return true_p








def main():
    mat_original = read_file('new.txt')
    mat_predicted = read_file('output.txt')
    mat_predicted_bin = binarize(mat_predicted, 0.3)
    [cor_pre, cor_org] = find_positive ( mat_original , mat_predicted_bin )
    [true_p] = true_positive(cor_pre, cor_org)
    #gnd_trth_mat = mat_cut(mat, 23, 183, 27, 91)
    contact = 1
    plt.imshow(mat_original)
    plt.show()
    plt.imshow(mat_predicted_bin)
    plt.show()
    #plt.title('Predicted contact map')
    #plt.ylim(233, 297)
    #plt.xlim(23, 182)
   # plt.show()



    print("done")

if __name__ == "__main__":
    main()



