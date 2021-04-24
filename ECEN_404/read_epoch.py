#this file does the data processing on the vector machine data to corporate with
#HRNN
import numpy as np
import time
from redundancy_elimination import DataManagement
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
        x_axis = []
        for line in lines:
            counter = 0
            if csv:
                words = line.split(",")
            else:
                words = line.split()

            x_axis .append(int(words[1]))
            train_loss.append(float(words[4]))
            val_loss.append(float(words[7]))
        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(x_axis, train_loss, label ='train_loss')
        plt.plot(x_axis, val_loss, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Avg loss')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(title)
        plt.show()



    """opens and reads a txt or csv file"""

    def open_file(self, filename, csv, low_lmt, hgh_lmt):
        if csv == 0:
            final_file = "iRefWeb_(" + str(low_lmt) + "_" + str(hgh_lmt) + ")"
            lines = []
            with open(filename, "r") as f:
                i = 0
                for line in f:
                    lines.append(line)
            self.fasta_link(lines, csv)


        else:
            lines = []
            with open(filename, "r") as f:
                i = 0
                for line in f:
                    lines.append(line)
            self.fasta_link(lines, csv)


    def process(self, filename):
        pro_list = []

        self.open_file(filename, csv=1, low_lmt=0, hgh_lmt=13000)

        print("done")

title = "task_4_l1_1e-4_stp-sz_1e-4"
def main():

    print("DONE")

    #task = DataManagementFasta4clmn(file_prot_pos='pos.txt', file_port_neg='negatome_seqnc_all.csv')
    task = DataManagementFasta4clmn()
    task.process("/home/argha/WORK/senior-design/ECEN_404/PPI_Xmodality/data_analysis/"+title+".csv")


if __name__ == "__main__":
    main()
    #testing()