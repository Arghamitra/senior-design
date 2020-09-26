import pandas
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import sklearn



def create_dca_matrix(name, len_d1, len_d2, option=True):
    df = pandas.read_csv(name, sep='\t',
                         header=None)  # loads the file, name, separated by tab, no header is given on the file
    df = df.sort_values([0, 1])  # there are three column in the file, first sorting column 0 and then column 1
    df = df[df[0] < len_d1 + 1]     # considering protein 1 to 40
    df = df[df[1] > len_d1]         #the rest of the protein
    matrix = np.array(df[2]).reshape(len_d1,
                                     len_d2)  # that means rearrange column 3 so that mum of row == len_d1 & num of column == len_d2

    if option == True:
        #plt.figure()
        plt.imshow(matrix)  # plots the matrix\
        #plt.savefig("pattern_comp1")
        plt.show()
        #plt.close()
    return matrix


def compute_matrix_result_for_one_filter(matrice_dca, mat_f):
    """
    Giving the dca matrix and a filter
    compute the correlation matrix
    """
    v = len(mat_f)      #the filter dimension
    v = v // 2
    (len_domain1, len_domain2) = matrice_dca.shape
    matrix_result = matrice_dca.copy()      #copies the entire dca matrix
    matrix_best_f = np.zeros((len_domain1, len_domain2))    #makes a zero matrix of the same dimension of matrice_dca
    for indice_1 in range(0, len_domain1):
        for indice_2 in range(0, len_domain2):      #this loop works as such: for protein 0 to all proteins form 0-104
            correlation = 0.0
            i_centre, j_centre = min(indice_1, v), min(indice_2, v)     #v is defined at top

            a = max(0, indice_1 - v)
            b = indice_1 + v + 1
            c = max(0, indice_2 - v)
            d = indice_2 + v + 1

            sous_matrix = matrice_dca[max(0, indice_1 - v):indice_1 + v + 1, max(0, indice_2 - v):indice_2 + v + 1]
            #the above line: taking the row/column of matrice_dca starting from maximum between 0 and indice_1-v
            #ending at indice_1+v+1
            mat_f[len(mat_f) // 2, len(mat_f) // 2] = np.nan        #the component of len(mat_f)//2 is made nan

            e = max(v - indice_1, 0)
            f = min(v * 2 + 1, len_domain1 - indice_1 + v)
            g = max(v - indice_2, 0)
            h = min(v * 2 + 1, len_domain2 + v - indice_2)

            m_f = mat_f[max(v - indice_1, 0):min(v * 2 + 1, len_domain1 - indice_1 + v),
                  max(v - indice_2, 0):min(v * 2 + 1, len_domain2 + v - indice_2)]      #takes a certain part of filter
            indic_flatt = m_f.shape[1] * i_centre + j_centre
            if matrice_dca[indice_1, indice_2] != sous_matrix.flatten()[indic_flatt]:   #flatten() converts a matrix in row vector
                print('Problem')
            ## remove the central value
            m_f = np.delete(m_f.flatten(), indic_flatt)     #we make the m_f a row vector deleting a component(indic_flatt)
            sous_matrix = np.delete(sous_matrix.flatten(), indic_flatt)
            # print scipy.stats.pearsonr(sous_matrix,m_f)[0]
            if np.sum(sous_matrix[0] == sous_matrix) != len(sous_matrix):
                correlation = scipy.stats.pearsonr(sous_matrix, m_f)[0]
            matrix_result[indice_1, indice_2] = correlation
    return matrix_result


def pattern_computation(v, dca_matrix, liste_mat_filtre):
    shape = dca_matrix.shape
    index = pandas.MultiIndex.from_product([range(s) for s in shape], names=list(
        'ij'))  # makes all possible indexes like (0 0) (0 1) ....(39 0)...(39 103)
    df = pandas.DataFrame({'dca': dca_matrix.flatten()},
                          index=index).reset_index()     # open df and dca, the output is very straight forward
    for num_mat, filter in enumerate(liste_mat_filtre):  # there are 6 filters in liste_mat_filtre
                                                        #here num_mat is the filter number and filter is the filter
        A = compute_matrix_result_for_one_filter(dca_matrix, filter)    #another function
        df['resul_{}_{}'.format(v, num_mat)] = A.flatten()  ## matrix result
    df['best_corr {}'.format(v)] = df[
        ['resul_{}_{}'.format(v, num_mat) for num_mat in range(len(liste_mat_filtre))]].max(axis=1)
    df['best_f {}'.format(v)] = df[
        ['resul_{}_{}'.format(v, num_mat) for num_mat in range(len(liste_mat_filtre))]].idxmax(axis=1)
    return df
