#list1 = ['1', '2', '3', '4']

s = ""

# joins elements of list1 by '-'
# and stores in sting s
#s = s.join(list1)
#print(s)

import numpy as np
#data = np.load('/home/at/work/data_processed/protein_train.npy')
#data1 = np.load('/home/at/work/data_processed/IC50_train.npy')

data1 = np.load('/home/at/work/dataset/ECEN_404_dataset/vector_machine_data/TEST_MID_positivesA_2021_02_14-23__18_26.npy')
sqnc = []
sqnc = sqnc+[1]*(len(data1))
matrix_1 = np.array(sqnc).reshape((len(data1),1))
np.save('TEST_positve_label', matrix_1)
print("DONE")