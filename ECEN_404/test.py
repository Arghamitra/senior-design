#list1 = ['1', '2', '3', '4']

s = ""

# joins elements of list1 by '-'
# and stores in sting s
#s = s.join(list1)
#print(s)
import numpy as np
from random import seed
import random as rand
sample_contact_map =[]

data1 = np.load('/home/argha/WORK/extracted_data/extracted_data/final_data/train_nA.npy')
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


"""
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

