






#opens the file where we have all the sequences
def open_sqnc_file(sqnc_filename):
    print("extract sequence")
    str_to_sqnc_dic = {}
    length = 0
    q = 0
    with open(sqnc_filename, "r") as f:
        i = 0
        for line in f:
            if (i % 2) == 0:
                token = line.strip('\n>')
            else:
                sqnc = line.strip('\n>')
                str_to_sqnc_dic[token] = sqnc
                length = length+len(sqnc)
                if len(sqnc)>2000:
                    q+=1
                #lines.append([token, line])
            i+=1
    avg_len = 2*length/i
    return str_to_sqnc_dic

def open_file(filename, sqnc_filename):

    str_to_sqnc_dic = open_sqnc_file(sqnc_filename)

def main():
    filename = "/home/at/work/dataset/ECEN_404_dataset/vector_machine_data/MID_HARD/TEST_MID_HARD_negatives.txt"
    sqnc_filename = "/home/argha/WORK/extracted_data/extracted_data/all_seq.fasta"
    open_file(filename, sqnc_filename )
    print("DONE")


if __name__ == "__main__":
    main()