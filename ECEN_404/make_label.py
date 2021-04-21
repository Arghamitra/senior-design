import numpy as np

def save_np_file(name, file):
    np.save(name, file)

def make_label(label, length):
    sqnc = []
    sqnc = sqnc + [label] * (length)
    matrix_1 = np.array(sqnc).reshape((length, 1))
    return matrix_1



TEST_HARD_P = make_label(1, len(np.load('/home/argha/WORK/extracted_data/extracted_data/2D_data/final_data/Test_sqnc_A_2021_04_07-16__19_46.npy')))
TEST_HARD_N = make_label(1, len(np.load('/home/argha/WORK/extracted_data/extracted_data/2D_data/final_data/VAL_sqnc_A_2021_04_06-02__53_12.npy')))

save_np_file('test_sqnc_lbl', TEST_HARD_P)
save_np_file('val_sqnc_lbl', TEST_HARD_N)