import urllib.request
#import urllib
import csv
from statistics import mean
import math
from matplotlib import pyplot as plt
import numpy as np
import time
import argparse

def write_outputfile(unq_pro_num, homodimers, avg_len, max_len, min_len, unq_int):
    print("write outputfile")
    trimester = time.strftime("_%Y_%m_%d-%H:%M:%S")
    filename = "output" + trimester + ".txt"
    file1 = open(filename, "w")
    str1 = ["The num of unique interactions is: " + str(len(unq_int)) +" \n",
            "The num of unique proteins is: " + str(unq_pro_num) +" \n",
           "The num of homodimers is: " + str(len(homodimers)) + " \n",
           "Avg len of protein sequences are: "+ str(round(avg_len)) +"\n",
           "Max leng of protein sequences are: "+ str(max_len) +"\n",
           "Min leng of protein sequences are: "+ str(min_len) + "\n",
           "\n" + "list of Homodimers" + " \n"]

    # \n is placed to indicate EOL (End of Line)
    file1.writelines(str1)
    for x in range(len(homodimers)):
        file1.write(str(homodimers[x])+ "\n")
    file1.close()  # to change file access modes

def draw_histogram(length_distribution):
    print("draw histogram")
    avg_len = mean(length_distribution)
    max_len = max(length_distribution)
    min_len = min(length_distribution)
    bin_interval = (max_len - min_len)/round(math.sqrt(len(length_distribution)))
    #bins = round(math.sqrt(len(length_distribution)))
    data = np.array(length_distribution)
    fig, axs = plt.subplots(figsize=(10, 7))
    axs.hist(data, bins = 400)
    plt.title('Protein sequence length distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.xlim(-100, 6000)
    plt.show()
    return (avg_len, max_len, min_len)


"""Prints out the headers and sequences in a final .csv file"""
def write_file( header_A, sequencs_A, header_B, sequencs_B, final_file, label):
    print("Write file")
    filename1 = "/home/at/work/ECEN_404/extracted_data/"
    trimester = time.strftime("_%Y_%m_%d-%H:%M:%S")
    filename = filename1 + final_file + trimester +".csv"
    length_distribution = []
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        for x in range(len(header_A)):
            row1 = [header_A[x], sequencs_A[x], header_B[x], sequencs_B[x],label]
            csvwriter.writerow(row1)
            length_distribution.append(len(sequencs_A[x]))
            length_distribution.append(len(sequencs_B[x]))
    return (length_distribution)




def fasta_link(lines, csv):
    length_distribution = []
    print("fasta_link")
    proA_links = []
    proB_links = []
    homodimers = []
    uniq_inte = []

    for line in lines:
        if csv:
            words = line.split(",")
        else:
            words = line.split()
        proA_links.append(words[0])
        proA_links.append(words[2])
        uniq_inte.append(min(words[0], words[2])+max(words[0], words[2]))
       # uniq_inte.append()

        length_distribution.append(len(words[1]))
        length_distribution.append(len(words[3]))
        if (words[1] == words[3]):
            homodimers.append([words[0], words[2]])
    pro_ID = (set(proA_links))
    uniq_inte = set(uniq_inte)
    return proA_links, proB_links, len(pro_ID), homodimers, length_distribution, uniq_inte

"""opens and reads a txt or csv file"""
def open_file(filename, csv, low_lmt, hgh_lmt):
    if csv == 0:
        pass

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
        proA_links, proB_link, unq_pro_num, homodimers, length_distribution, uniq_inte = fasta_link(lines, csv)
        avg_len, max_len, min_len = draw_histogram(length_distribution)
        write_outputfile(unq_pro_num, homodimers, avg_len, max_len, min_len, uniq_inte)

def main():
    """
    parser = argparse.ArgumentParser(description= 'Required parameters to run the script')
    parser.add_argument('-ll','--low_lmt',type=int, help ='first line to start fasta extraction')
    parser.add_argument('-hl','--hgh_lmt',type=int, help ='last line to end fasta extraction')
    args = parser.parse_args()
    """



    open_file(filename="/home/argha/WORK/extracted_data/extracted_data/negatome_seqnc_all.csv", csv = 1, low_lmt=0, hgh_lmt=320000)
    open_file(filename="/home/argha/WORK/extracted_data/extracted_data/iRefWeb_all.csv", csv=1, low_lmt=0,
              hgh_lmt=320000)
    print("DONE")


if __name__ == "__main__":
    main()
