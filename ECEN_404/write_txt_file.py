
import numpy as np
import time

def write_file(file_npy, final_file_name):

    file1 = open(final_file_name,"w")

    for couples in file_npy:
        file1.write(couples[0] +'  '+couples[1]+'\n')
    file1.close()


def main():
    file_npy = np.load("FINAL_inter2D_list_proLenLess1000.npy")
    write_file(file_npy, "FINAL_inter2D_list_proLenLess1000.txt")
    print("DONE")


if __name__ == "__main__":
    main()