#list1 = ['1', '2', '3', '4']

s = ""

# joins elements of list1 by '-'
# and stores in sting s
#s = s.join(list1)
#print(s)
import numpy as np

#data = np.load('/home/argha/WORK/extracted_data/vector_machine_data/protein_train.npy')
#data1 = np.load('/home/at/work/data_processed/IC50_train.npy')

#data1 = np.load('/home/argha/WORK/extracted_data/vector_machine_data/VAL_EASY_negativesA_2021_02_11-03__51_26.npy')
data1 = np.load('negative_list.npy')
sqnc = []
sqnc = sqnc+[1]*(len(data1))
matrix_1 = np.array(sqnc).reshape((len(data1),1))
np.save('TEST_positve_label', matrix_1)
print("DONE")
"""



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

