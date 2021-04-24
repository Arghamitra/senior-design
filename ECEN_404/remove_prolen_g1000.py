import numpy as np
import time
import glob
import os

def cut_sqnc_bgr_1000(proA, proB, str_to_sqnc_dic):

    if (str_to_sqnc_dic[proA]>1000) and (str_to_sqnc_dic[proB]>1000):
        print("both bigger than 1000")
        return ('del')
    elif str_to_sqnc_dic[proB]>1000:
        print("B is bigger")
        return ('del')
    elif str_to_sqnc_dic[proA]>1000:
        print("B is bigger")
        return ('del')
    elif proA == 'P20226':
        print("Bad interaction")
        return ('del')

    else:
        return ('ok')


# opens the file where we have all the sequences and makes a dictionary
def open_sqnc_file(sqnc_filename):
    print("extract sequence")
    str_to_sqnc_dic = {}
    with open(sqnc_filename, "r") as f:
        i = 0
        for line in f:
            if (i % 2) == 0:
                token = line.strip('\n>')
            else:
                sqnc = line.strip('\n>')
                str_to_sqnc_dic[token] = len(sqnc)
                # lines.append([token, line])
            i += 1
    return str_to_sqnc_dic


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
        data_A.append(split_data[0])
    return data_A, data_B, data_touple

def open_file(filename, sqnc_filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    data_A, data_B, data_touple = extract_id(lines)
    str_to_sqnc_dic = open_sqnc_file(sqnc_filename)
    Final_list =[]

    for proA, proB in zip(data_A, data_B):
        dcsn = cut_sqnc_bgr_1000(proA, proB, str_to_sqnc_dic)
        if (dcsn != 'del'):
            Final_list.append([proA, proB])
    Final_list = np.array(Final_list)
    np.save("FINAL_inter2D_list_proLenLess1000", Final_list)

def main():

    #file_npy = np.load("/home/argha/WORK/extracted_data/extracted_data/positive_list.npy")
    filename = "/home/argha/WORK/extracted_data/extracted_data/2D_data/overlaps_68.txt"
    sqnc_filename = "/home/argha/WORK/extracted_data/extracted_data/all_seq.fasta"
    open_file(filename, sqnc_filename )
    print("DONE")


if __name__ == "__main__":
    main()