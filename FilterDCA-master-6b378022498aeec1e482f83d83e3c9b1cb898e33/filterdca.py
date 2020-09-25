import pandas
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import sklearn
from fdca_helper import *





                                ####### the main function started ########

name = 'combined_MSA_ddi_3_PF10417_PF00085_result'
lend1, lend2 = 40, 104
dca_matrix = create_dca_matrix(name, lend1, lend2, option=True)


## load the 6 filters of the selected size
v: int = 69
liste_mat_filtre = list(np.load('maps/{}/list_mat.npy'.format(
    v)))  # goes to the maps folder and finds out the npy file of given filter (in this case filter 69)

## Apply each of the 6 filters on the dca matrix
df = pattern_computation(v, dca_matrix, liste_mat_filtre)

correlation_matrix = np.array(df['best_corr {}'.format(v)]).reshape((dca_matrix.shape))
plt.imshow(correlation_matrix)
plt.title('Pattern score')
plt.colorbar()
plt.show()

# Load the classifiar and the min and max values for the pattern score
size_meff = 'big'
clf = pickle.load(open('classifier/{}-{}-linear-clf.sav'.format(v, size_meff), "rb"), encoding='latin1')
min_c, max_c = np.loadtxt('classifier/min_max_{}_{}'.format(v, size_meff))

# We normalize the 'best corr' variable
df['corr {}'.format(v)] = (df['best_corr {}'.format(v)] - min_c) / (max_c - min_c)

column = ['dca', 'corr {}'.format(v)]
X = np.array(df[column])
probability = clf.predict_proba(X)[:, 1]
df['proba contact'] = probability

plt.imshow(probability.reshape(dca_matrix.shape))
plt.title('Probability of being a contact')
plt.colorbar()
plt.show()

contact = probability > 0.3
plt.imshow(contact.reshape(dca_matrix.shape))
plt.title('Predicted contact map')
plt.show()

## to save results
df.to_csv('results.dat', index=False)


                ###python GUI###


