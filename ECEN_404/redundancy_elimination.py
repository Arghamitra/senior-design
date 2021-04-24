import urllib.request
import os
#import urllib
import csv
from statistics import mean
import math
from matplotlib import pyplot as plt
import numpy as np
import time
import argparse
import sys

#this function removes all the uncommon protein pairs
def remove_uncommon_pro(uncommon_pro, pro_id_pair):
    new_list = []
    new_pair =[]
    for A, B in pro_id_pair:
        if (A != uncommon_pro) and (B!= uncommon_pro):
            new_pair.append([A, B])
            new_list.append(A)
            new_list.append(B)
    return new_list, new_pair

#form list A and list B, this function returns list C = element(A) - element (B)
def comprehension(a, b):
    c =  [x for x in a if x not in b]
    return c


#finds out and unique elements of two list
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    c = (a_set & b_set)
    return a,b,c


"""finds out the common and uncommon portiens from two list"""
def common_uncommon(p_list, n_list):
    print("common uncommon")
    p_list , n_list, common_pro = common_member(p_list , n_list)
    Total_list = p_list + n_list
    uncommon_pro =  comprehension(Total_list, common_pro)
    return common_pro, uncommon_pro

class DataManagement():


    def __init__(self, name):
        name = name

    """
        def __init__(self, file_prot_pos, file_port_neg):
        self.file_prt_pos = file_prot_pos
        self.file_prt_neg = file_port_neg

        self.path_prt_pos = self.make_path(self.file_prt_pos)
        self.path_prt_neg = self.make_path(self.file_prt_neg)


    """
    def convertFASTA(self, filename, dictionary, pro_list):

        fasta_file = open(filename, "w")
        for protein in pro_list:
            fasta_file.write(">" + str(protein) + "\n")
            fasta_file.write(dictionary[protein] + "\n")
        fasta_file.close()


    def make_path(self, file_name):
        #dir_path = os.getcwd() + "/../../extracted_data/extracted_data/"
        dir_path = os.getcwd() 
        path = os.path.join(dir_path, file_name)
        return path

    """"reads lines and returns a fasta link/ protein id"""
    def fasta_link(self, lines, csv):
        print("fasta_link")
        pro_id =[]
        pro_id_pair = []
        homodimers = []
        for line in lines:
            if csv:
                words = line.split(",")
            else:
                words = line.split()
            try:
                protienA_id = words[1]
                protienB_id = words[2].strip("\n")
            except:
                print("errror to find the ID ", words[0])
            if '_' in protienA_id or '_' in protienB_id or (protienA_id.isalpha()) or (protienB_id.isalpha()):
                continue
            else:
                try:
                    pro_id_pair.append([protienA_id, protienB_id] )

                    pro_id.append(protienA_id)
                    pro_id.append(protienB_id)
                    if(protienA_id == protienB_id):
                        homodimers.append([protienA_id, protienB_id])

                except:
                    print("Error to make link: ", protienA_id, protienB_id)


        return pro_id, pro_id_pair


    """opens and reads a txt or csv file"""
    def open_file(self, filename, csv, low_lmt, hgh_lmt):
        if csv == 0:
            final_file = "iRefWeb_(" + str(low_lmt) +"_"+ str(hgh_lmt)+")"
            lines = []
            with open(filename, "r") as f:
                i = 0
                for line in f:
                    if (i >= low_lmt):
                        lines.append(line)
                    i += 1
                    if (i == hgh_lmt):
                        break
            list, pro_id_pair = self.fasta_link(lines, csv)


        else:
            lines = []
            with open (filename, "r") as f:
                i = 0
                for line in f:
                    if (i >= low_lmt):
                        lines.append(line)
                    i += 1
                    if (i == hgh_lmt):
                        break
            list, pro_id_pair = self.fasta_link(lines, csv)

        return (list, pro_id_pair)

    def process(self, filename):
        """

        n_list, pro_id_pair_n = self.open_file(filename=self.path_prt_pos,
                                          csv=1, low_lmt=0, hgh_lmt=13000)
        p_list, pro_id_pair_p = self.open_file(filename=self.path_prt_neg,
                                          csv=0, low_lmt=0, hgh_lmt=400000)

        """
        n_list, pro_id_pair_n = self.open_file(filename = self.make_path(filename),
                                               csv=1, low_lmt=0, hgh_lmt=13000)
        p_list, pro_id_pair_p = self.open_file(filename = self.make_path(filename),
                                               csv=0, low_lmt=0, hgh_lmt=400000)

        counter = 0;

        common_id, uncommon_id = common_uncommon(p_list, n_list)

        # can be modularized
        while (len(common_id) > 0) and (len(uncommon_id) > 0):
            print("num common", len(common_id), "num uncommon", len(uncommon_id))
            n_list, pro_id_pair_n = remove_uncommon_pro(uncommon_id[0], pro_id_pair_n)
            p_list, pro_id_pair_p = remove_uncommon_pro(uncommon_id[0], pro_id_pair_p)
            uncommon_id.pop(0)
            common_id, uncommon_id = common_uncommon(p_list, n_list)
        print("DONE")
        np.save("positive_list", pro_id_pair_p)
        np.save("negative_list", pro_id_pair_n)




def main():
    """
    parser = argparse.ArgumentParser(description= 'Required parameters to run the script')
    parser.add_argument('-ll','--low_lmt',type=int, help ='first line to start fasta extraction')
    parser.add_argument('-hl','--hgh_lmt',type=int, help ='last line to end fasta extraction')
    args = parser.parse_args()
    """

    task = DataManagement(name = 'real_data')
    task.process()




if __name__ == "__main__":
    main()
