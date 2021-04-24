#this file does the data processing on the vector machine data to corporate with
#HRNN
import numpy as np
import time
from redundancy_elimination import DataManagement


class DataManagementFasta4clmn(DataManagement):

    def __init__(self, name= ''):
        name = name
    """
    def __init__(self, file_prot_pos, file_port_neg):
        super().__init__(file_prot_pos=file_prot_pos, file_port_neg=file_port_neg)
    """

    def fasta_link(self, lines, csv):
        print("fasta_link")
        pro_id = []
        id_seq_dic = dict()
        for line in lines:
            if csv:
                words = line.split(",")
            else:
                words = line.split()
            try:
                protienA_id = words[0]
                protienB_id = words[2]
                protienA_seq = words[1]
                protienB_seq = words[3]
            except:
                print("errror to find the ID ", words[0])
            if '_' in protienA_id or '_' in protienB_id or (protienA_id.isalpha()) or (protienB_id.isalpha()):
                continue
            else:
                try:
                    id_seq_dic[protienA_id] = protienA_seq
                    id_seq_dic[protienB_id] = protienB_seq

                except:
                    print("Error to make link: ", protienA_id, protienB_id)

        return id_seq_dic

    """opens and reads a txt or csv file"""

    def open_file(self, filename, csv, low_lmt, hgh_lmt):
        if csv == 0:
            final_file = "iRefWeb_(" + str(low_lmt) + "_" + str(hgh_lmt) + ")"
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
            with open(filename, "r") as f:
                i = 0
                for line in f:
                    if (i >= low_lmt):
                        lines.append(line)
                    i += 1
                    if (i == hgh_lmt):
                        break
            id_seq_dic = self.fasta_link(lines, csv)

        return (id_seq_dic)

    def process(self, filename):
        pro_list = []

        id_seq_dic = self.open_file(filename=self.make_path(filename),
                                               csv=0, low_lmt=0, hgh_lmt=13000)
        pro_pair = np.load('negative_list.npy')

        for pair in pro_pair:
            pro_list.append(pair[0])
            pro_list.append(pair[1])

        self.convertFASTA("all_seq.fasta", id_seq_dic, set(pro_list))
        print("done")


def main():

    print("DONE")

    #task = DataManagementFasta4clmn(file_prot_pos='pos.txt', file_port_neg='negatome_seqnc_all.csv')
    task = DataManagementFasta4clmn()
    task.process('Example1Out_1_7034651.txt')


if __name__ == "__main__":
    main()
    #testing()