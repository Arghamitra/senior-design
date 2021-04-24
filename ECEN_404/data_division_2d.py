import numpy as np
import time
import random

#saves in np form
def np_save(train_set, test_set, val_set):

    np.save('Train_intrctn_list', train_set)
    np.save('Test_intrctn_list', test_set)
    np.save('Val_intrctn_list', val_set)

def split_data(data, train_offset, test_offset):
    random.shuffle(data)

    train_data = data[:train_offset]
    val_data = data[(train_offset): (train_offset + test_offset)]
    test_data = data[(train_offset + test_offset): ]
    return train_data, test_data, val_data

#extract protein id from text file
def extract_id(lines):
    print("id")
    data_A = []
    data_B = []
    data_touple =[]
    for line in lines:
        split_data = str(line).split()
        data_touple.append([split_data[0], split_data[1]])
        data_B.append(split_data[1])
        data_B.append(split_data[0])
    return data_A, data_B, data_touple

def open_file(filename, sqnc_filename):
    #text_file = open(filename, "r")
    #lines = text_file.readlines()
    #data_A, data_B, data_touple = extract_id(lines)
    data_touple = []
    for couples in filename:
        data_touple.append([couples[0], couples[1]])
    train_set, test_set, val_set = split_data(data_touple, train_offset = 30, test_offset = 8)
    np_save(train_set, test_set, val_set)
    return data_touple



def main():

    file_npy = np.load("FINAL_inter2D_list_proLenLess1000.npy")
    #filename = "/home/argha/WORK/extracted_data/extracted_data/2D_data/overlaps_68.txt"
    sqnc_filename = "/home/at/work/dataset/ECEN_404_dataset/vector_machine_data/sequences.fasta"
    #filename = "/home/at/work/dataset/ECEN_404_dataset/vector_machine_data/MID_HARD/TEST_MID_HARD_negatives.txt"
    #sqnc_filename = "/home/at/work/dataset/ECEN_404_dataset/vector_machine_data/sequences.fasta"

    data_touple = open_file(file_npy, sqnc_filename )
    print("DONE")


if __name__ == "__main__":
    main()