#this file does the data processing on the vector machine data to corporate with
#HRNN
import numpy as np
import time
from data_division import DataManagement
import matplotlib.pyplot as plt


class DataManagementFasta4clmn(DataManagement):

    def __init__(self, name= ''):
        name = name
    """
    def __init__(self, file_prot_pos, file_port_neg):
        super().__init__(file_prot_pos=file_prot_pos, file_port_neg=file_port_neg)
    """

    def fasta_link(self, lines, csv):
        print("fasta_link")
        train_loss =[]
        val_loss =[]
        for line in lines:
            counter = 0
            if csv:
                words = line.split(",")
            else:
                words = line.split()

            if words[2] == 'train':
                try:
                    loss1 = float(words[4])
                    loss2 = float(words[7])
                except:
                    print("errror to find the ID ", words[0])
                train_loss.append(loss1)
                val_loss.append(loss2)
        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(train_loss, label ='train_loss')
        plt.plot(val_loss, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Avg loss')
        plt.legend()
        plt.grid()
        plt.show()



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
            self.fasta_link(lines, csv)


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

        print("done")


def main():

    print("DONE")

    #task = DataManagementFasta4clmn(file_prot_pos='pos.txt', file_port_neg='negatome_seqnc_all.csv')
    task = DataManagementFasta4clmn()
    task.process('Example1Out_1_7034651.txt')


if __name__ == "__main__":
    main()
    #testing()