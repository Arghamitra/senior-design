import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Fixing random state for reproducibility
np.random.seed(19680801)

N_points = 100000
n_bins = 20

# Generate a normal distribution, center at x=0 and y=5
x = np.random.randn(N_points)
y = .4 * x + np.random.randn(100000) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(x, bins=n_bins)
axs[1].hist(y, bins=n_bins)
plt.show()

""""reads lines and returns a fasta link from the uniprod id"""
x = 'P38276'
y = 'P38276'


labels =[[1,2,3],[4,5,6]]
matrix_1 = np.array(labels).reshape((2,3))
trth = matrix_1.reshape(3 * 2)


#list1 = ['1', '2', '3', '4']

s = ""

# joins elements of list1 by '-'
# and stores in sting s
#s = s.join(list1)
#print(s)
import torch
import torch.nn as nn
import numpy as np
from random import seed
import random as rand
from sklearn.metrics import roc_curve, auc, average_precision_score

file1 = open("myfile.txt","w")
L = ["This is Delhi \n","This is Paris \n","This is London \n"]
file1.write(L)
file1.close()


data1 = np.load('Intra_train_A.npy')
i=2

"""
n = 4
a = [[rand.randint(0, 10) for _ in range(n)] for _ in range(n)]
a = np.array(a).reshape(2, 8)
a = a.transpose()

labels =[[1,2,3],[4,5,6]]
y_pred =[]
ap =[]
for x in range(1):
    seed()
    #k = rand.randint(0, 1) # decide on k once
    #for _ in range(10):
        #print(k) # print k over and over again

    n = 6
    a = [[rand.randint(0, 1) for _ in range(n)] for _ in range(n)]
    a = np.array(a).reshape(6,6)
    labels.append(a)
labels = np.array(labels)

for x in range(4):
    seed()
    #k = rand.randint(0, 1) # decide on k once
    #for _ in range(10):
        #print(k) # print k over and over again

    n = 6
    a = [[rand.randint(0, 1) for _ in range(n)] for _ in range(n)]
    a = np.array(a).reshape(6,6)
    y_pred.append(a)

y_pred = np.array(y_pred)

for ix in range(np.shape(labels)[0]):
    fpr_whole, tpr_whole, _ = roc_curve(labels[ix], y_pred[ix])
    average_precision_whole = average_precision_score(labels[ix], y_pred[ix])
    ap.append(average_precision_whole)



def padding_masking(INTER_predctn, len_prot_data1, len_prot_data2):
    ##joint_attn = torch.einsum('bij,bi,bj->bij', joint_attn, prot_mask, comp_mask)

    inter_prot_prot_masked = torch.empty(INTER_predctn.size()[1], INTER_predctn.size()[1])
    for ix in range(len(len_prot_data1)):
        contact_map = INTER_predctn[ix]

        vector1 = []
        vector1 = vector1 + [1] * (len_prot_data1[ix])
        vector1 = vector1 + [0] * (contact_map.size()[0] - len_prot_data1[ix])
        prot_mask1 = torch.tensor(np.array(vector1))

        vector2 = []
        vector2 = vector2 + [1] * (len_prot_data2[ix])
        vector2 = vector2 + [0] * (contact_map.size()[0] - len_prot_data2[ix])
        prot_mask2 = torch.tensor(np.array(vector2))

        contact_map = torch.einsum('ij,i,j->ij', contact_map, prot_mask2, prot_mask1)

        if (ix == 0):
            inter_prot_prot_masked = torch.cat([contact_map.unsqueeze(0)], dim=0)
        else:
            inter_prot_prot_masked = torch.cat([inter_prot_prot_masked, contact_map.unsqueeze(0)], dim=0)
    return inter_prot_prot_masked

INTER_predctn = torch.rand(3, 2, 6)
len_prot_data1 = [1,2,3]
len_prot_data2 = [4, 5, 6]
padding_masking(INTER_predctn, len_prot_data1, len_prot_data2)


sample_contact_map =[]

sqnc = [1,2,3,0,0,0,0, 4]
X = np.array(sqnc)
#X = np.random.randn(1e3, 5)
X[np.abs(X)< .1]= 0 # some zeros
#X = np.ma.masked_equal(X,0)
Y = X[X != 0]

sqnc = [1,2,3,0,0,0,0]
sqnc = np.array(sqnc)


data1 = np.load('/home/argha/WORK/extracted_data/extracted_data/2D_data/ctmaps_inter/ctmap_1a02_P05412_Q13469.npy')

for i in range(data1.shape[0]):
    for j in range(data1.shape[1]):
        if data1[i][j] == 1:
            print("there is one")

data2 = np.load('/home/argha/WORK/extracted_data/extracted_data/final_data/train_pA.npy')
data3 = np.load('L_val_nA.npy')

for x in range(4):
    seed()
    #k = rand.randint(0, 1) # decide on k once
    #for _ in range(10):
        #print(k) # print k over and over again

    n = 6
    a = [[rand.randint(0, 1) for _ in range(n)] for _ in range(n)]
    a = np.array(a).reshape(6,6)
    sample_contact_map.append(a)

sample_contact_map = np.array(sample_contact_map)
np.save('sample_contact_map', sample_contact_map)

#
num_homo_neg = 0
data1 = np.load('/home/argha/WORK/extracted_data/ctmaps/ctmap_1a4y_P03950_P13489.npy')
data2 = np.load('/home/argha/WORK/extracted_data/vector_machine_data/L_train_pB.npy')



for interaction in data1:
    protein1 = interaction[0]
    protein2 = interaction[1]
    if (protein1 == protein2):
        num_homo_neg+=1

num_homo_pos = 0
data2 = np.load('/home/argha/WORK/extracted_data/extracted_data/positive_list.npy')
for interaction in data2:
    protein1 = interaction[0]
    protein2 = interaction[1]
    if (protein1 == protein2):
        num_homo_pos+=1

sqnc = []
sqnc = sqnc+[1]*(len(data1))
matrix_1 = np.array(sqnc).reshape((len(data1),1))
np.save('TEST_positve_label', matrix_1)
print("DONE")



def common_member(a, b):
    a_set = set(a)
    b_set = set(b)

    
        if (a_set & b_set):
        print(a_set & b_set)
    else:
        print("No common elements")

    
    c = (a_set & b_set)
    return c




def comprehension(a, b):
    return [x for x in a if x not in b]


def main():
    a = [1, 2, 3, 4, 5]
    b = [5, 6, 7, 8, 9, 10, 11]
    c = common_member(a, b)

    a = [1, 2, 3, 4, 5]
    b = [6, 7, 8, 9]
    common_member(a, b)

    a = ['a','b','c','d','e']  # Source list
    b = ['a']  # Items to remove
    c = comprehension(a, b)
    print("DONE")


if __name__ == "__main__":
    main()
"""

