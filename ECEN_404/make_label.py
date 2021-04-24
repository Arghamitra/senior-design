import numpy as np

def save_np_file(name, file):
    np.save(name, file)

def make_label(label, length):
    sqnc = []
    sqnc = sqnc + [label] * (length)
    matrix_1 = np.array(sqnc).reshape((length, 1))
    return matrix_1



TEST_HARD_P = make_label(1, len(np.load('train_sqnc_A_2021_04_22-22__03_26.npy')))
#TEST_HARD_N = make_label(1, len(np.load('val_sqnc_A_2021_04_22-22__02_21.npy')))

save_np_file('train_sqnc_lbl', TEST_HARD_P)
#save_np_file('val_sqnc_lbl', TEST_HARD_N)