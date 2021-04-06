import numpy as np

#data = np.load('/home/argha/WORK/extracted_data/vector_machine_data/protein_train.npy')
#data1 = np.load('/home/at/work/data_processed/IC50_train.npy')

#data1 = np.load('/home/argha/WORK/extracted_data/vector_machine_data/VAL_EASY_negativesA_2021_02_11-03__51_26.npy')

def split_mat(data, offset):
    A = data[:offset, :, :]
    B = data[offset:, :, :]
    return A, B

def make_label(label, length):
    sqnc = []
    sqnc = sqnc + [label] * (length)
    matrix_1 = np.array(sqnc).reshape((length, 1))
    return matrix_1

def save_np_file(name, file):
    np.save(name, file)

num_homo_neg = 0
data1 = np.load('train_pA.npy')
data2 = np.load('train_nA.npy')
offset1 = int(data1.shape[0]*0.1)
offset2 = int(data2.shape[0]*0.1)
train_pB, val_pB = split_mat(data1, offset1)
train_nB, val_nB = split_mat(data2,offset2)

L_train_pB = make_label(1, len(train_pB))
L_val_pB = make_label(1, len(val_pB))
L_train_nB = make_label(0, len(train_nB))
L_val_nB = make_label(0, len(val_nB))

save_np_file('test_pA', train_pB)
save_np_file('test_nA', train_nB)
save_np_file('train_pA', val_pB)
save_np_file('train_nA', val_nB)

save_np_file('L_test_pA', L_train_pB)
save_np_file('L_test_nA', L_train_nB)
save_np_file('L_train_pA', L_val_pB)
save_np_file('L_train_nA', L_val_nB)