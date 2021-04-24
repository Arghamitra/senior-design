import numpy as np
import time
import random
import os
import sys
import zipfile
import shutil

#saves in np form
def np_save(INTRA_train_A,  INTRA_train_B, setA_file , setB_file):

    np.save(setA_file, INTRA_train_A)
    np.save(setB_file, INTRA_train_B)

#the formation of individual matrix
def makt_contact_map(pos_A, pos_B, value, max_len):
    sqnc = [0]*(max_len*max_len)
    matrix_1 = np.array(sqnc).reshape((max_len, max_len))
    for i in range(len(pos_A)):
        if pos_A[i] >= 1000 or pos_B[i] >= 1000:
            pass
        else:
            matrix_1[pos_A[i]][pos_B[i]] = value[i]
    return matrix_1

#extract protein id from text file
def extract_contact(lines):
    print("id")
    pos_A = []
    pos_B = []
    value =[]
    for line in lines:
        split_data = str(line).split()
        if split_data[0].isnumeric():
            pos_A.append(int(split_data[0]))
            pos_B.append(int(split_data[1]))
            value.append(1 if float(split_data[2]) > 0.5 else 0)

        else:
            print()
    return pos_A, pos_B, value

#reads the file line by line
def open_file(filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    pos_A, pos_B, value = extract_contact(lines)
    contact_map = makt_contact_map(pos_A, pos_B, value, max_len = 1000)
    return contact_map

#parsing file name
def read_file_name(root_path):

    files = os.listdir(root_path)

    # filter out only zip files
    files = [file for file in files if file.endswith('.txt')]
    File_name = dict()

    #splits file name and make dictionary with uniprod protein
    for file in files:
        split_data = str(file).split('.')
        File_name[split_data[0]] = file
    return File_name

def file_select(path, File_name, cntct_mp_files_path):
    protein_list = np.load(path)
    INTRA_train_A = []
    INTRA_train_B = []

    #finds all the list of train/test/val set
    for proteins in protein_list:

        #finds the name of file for that protien using dictionary
        proteinA = File_name[proteins[0]]
        proteinB = File_name[proteins[1]]

        #creats file root and gets the contact map for that protein
        contact_mapA = open_file(filename = cntct_mp_files_path + '/' + str(proteinA))
        contact_mapB = open_file(filename=cntct_mp_files_path + '/' + str(proteinB))

        INTRA_train_A.append(contact_mapA)
        INTRA_train_B.append(contact_mapB)

    #final list of contact for train/test/val set
    INTRA_train_A = np.array(INTRA_train_A)
    INTRA_train_B = np.array(INTRA_train_B)

    #saves file
    np_save(INTRA_train_A,  INTRA_train_B, setA_file = 'Intra_train_A', setB_file = 'Intra_train_B')



def main():

    cntct_mp_files_path = '/home/argha/WORK/extracted_data/extracted_data/2D_data/INTRA_cntct_map'
    path='Train_intrctn_list.npy'
    File_name = read_file_name(cntct_mp_files_path)
    file_select(path, File_name, cntct_mp_files_path)

    print("DONE")


if __name__ == "__main__":
    main()

