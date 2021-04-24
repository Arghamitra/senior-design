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
    else:
        return ('ok')



def make_final_matrix(matrix, proA, proB, str_to_sqnc_dic):
    lenA = str_to_sqnc_dic[proA]
    lenB = str_to_sqnc_dic[proB]

    if (str_to_sqnc_dic[proA]>1000) and (str_to_sqnc_dic[proB]>1000):
        matrix_1 = matrix[lenA:(lenA+1000), :1000]
        print("both bigger than 1000")
    elif str_to_sqnc_dic[proB]>1000:
        matrix_1 = matrix[lenA:(lenA+1000), :(lenA)]
        print("B is bigger")
    elif str_to_sqnc_dic[proA]>1000:
        matrix_1 = matrix[lenA:, :1000]
        print("A is bigger")

    else:
        matrix_1 = matrix[lenA:, :(lenA)]

    matrix_1 = matrix_1.transpose()
    padded_matrix = np.zeros((1000, 1000))
    padded_matrix[:matrix_1.shape[0], :matrix_1.shape[1]] = matrix_1
    # padded_matrix = padded_matrix.transpose()
    total_len = (1000 * 1000)

    pos_val_1 =0
    for row in range(lenA):
        for clmn in range (lenB):
            try:
                if (padded_matrix [row][clmn] == 1):
                    pos_val_1 +=1
                    print(proB,' ', proA, ' ', row, ' ', clmn)
            except:
                pass



    pos_val_1 = pos_val_1/total_len
    return padded_matrix, pos_val_1


def find_contact_file( proA, proB, INTER_cntct_path):
    #os.chdir(INTER_cntct_path)
    matrix = np.zeros((1,1))
    i = 0
    try:
        for files in glob.glob(INTER_cntct_path +'ctmap_*_'+ proA +'_'+ proB +'.npy'):
            i+=1
            matrix = np.load(files)
            if (i>1):
                print("more than a file found", proA, '+', proB)
    except:
        print("could not find "+ proA +'_'+ proB)

    return matrix



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


def open_file(file_npy, sqnc_filename):
    data_A =[]
    data_B = []
    for couples in file_npy:
        data_A.append(couples[0])
        data_B.append(couples[1])

    str_to_sqnc_dic = open_sqnc_file(sqnc_filename)
    all_INTER_map = []
    Final_list =[]
    i = 0
    pos_val_1_l = []
    total_len_l = []
    for proA, proB in zip(data_A, data_B):
        matrix = find_contact_file(proA, proB, INTER_cntct_path = '/home/argha/WORK/extracted_data/extracted_data/2D_data/ctmaps/')
        """
        dcsn = cut_sqnc_bgr_1000(proA, proB, str_to_sqnc_dic)
        if (dcsn != 'del'):
            Final_list.append([proA, proB])
        i+=1
    Final_list = np.array(Final_list)
    np.save("Val_intrctn_listF", Final_list)

        """
        i += 1
        print(proA, '+', proB, ' ', i)
        padded_matrix, pos_val_1  = make_final_matrix(matrix, proA, proB, str_to_sqnc_dic)
        all_INTER_map.append(padded_matrix)
        pos_val_1_l.append(pos_val_1)


    np.array(all_INTER_map)
    print ("average +ve interaction", sum(pos_val_1_l)/len(pos_val_1_l))
    #np.save("Test_INTER_map", all_INTER_map)


def main():
    file_npy = np.load("Train_intrctn_list.npy")
    sqnc_filename = "/home/argha/WORK/extracted_data/extracted_data/all_seq.fasta"
    # filename = "/home/at/work/dataset/ECEN_404_dataset/vector_machine_data/MID_HARD/TEST_MID_HARD_negatives.txt"
    # sqnc_filename = "/home/at/work/dataset/ECEN_404_dataset/vector_machine_data/sequences.fasta"
    open_file(file_npy, sqnc_filename)
    print("DONE")


if __name__ == "__main__":
    main()