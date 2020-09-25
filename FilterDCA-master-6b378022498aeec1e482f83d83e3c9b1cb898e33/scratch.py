import tkinter as tk
from fdca_helper import *
import pandas
from PIL import ImageTk, Image
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import sklearn


def generate_dca_matrix(pic_opt=True):
    name = 'combined_MSA_ddi_3_PF10417_PF00085_result'
    lend1, lend2 = 40, 104
    dca_matrix = create_dca_matrix(name, lend1, lend2, option=pic_opt)
    return dca_matrix, name, lend1, lend2


def pattern_comp():  ## load the 6 filters of the selected size
    v: int = 69
    liste_mat_filtre = list(np.load('maps/{}/list_mat.npy'.format(
        v)))  # goes to the maps folder and finds out the npy file of given filter (in this case filter 69)

    dca_matrix, name, lend1, lend2 = generate_dca_matrix(pic_opt=False)
    ## Apply each of the 6 filters on the dca matrix
    df = pattern_computation(v, dca_matrix, liste_mat_filtre)

    correlation_matrix = np.array(df['best_corr {}'.format(v)]).reshape((dca_matrix.shape))
    plt.figure(1)
    plt.imshow(correlation_matrix)
    plt.title('Pattern score')
    plt.colorbar()
    #plt.show()
    #plt.savefig("pattern_comp2")

    # Load the classifiar and the min and max values for the pattern score
    size_meff = 'big'
    clf = pickle.load(open('classifier/{}-{}-linear-clf.sav'.format(v, size_meff), "rb"), encoding='latin1')
    min_c, max_c = np.loadtxt('classifier/min_max_{}_{}'.format(v, size_meff))

    # We normalize the 'best corr' variable
    df['corr {}'.format(v)] = (df['best_corr {}'.format(v)] - min_c) / (max_c - min_c)

    column = ['dca', 'corr {}'.format(v)]
    X = np.array(df[column])
    probability = clf.predict_proba(X)[:, 1]
    df['proba contact'] = probability

    contact = probability > 0.3
    plt.figure(2)
    plt.imshow(contact.reshape(dca_matrix.shape))
    plt.title('Predicted contact map')
    plt.show()
   # plt.savefig("pattern_comp3")

    ## to save results
    df.to_csv('results.dat', index=False)

    ###############GUI#############


root = tk.Tk()
label = tk.Label(root, text = "filterDCA Input")
label.pack(side = "top")

#input canvas
canvas = tk.Canvas (root, relief = "raised", borderwidth = 1)
canvas.pack(side = 'top', fill = tk.X)
img = ImageTk.PhotoImage(Image.open("protein1.png"))
canvas.create_image(950, 50, image=img)
img2 = ImageTk.PhotoImage(Image.open("protein2.png"))
canvas.create_image(950, 200,  image=img2)

#button 1
leftFrame = tk.LabelFrame(root, text = "filterDCA Functions", padx = 100, pady = 0)
leftFrame.pack(padx=10, pady=10)
b1 = tk.Button(leftFrame, text="DCA Matrix", command=generate_dca_matrix
               , width=20, height=4)
b1.pack( side="top")
b1.pack(padx=0, pady=0)


b2 = tk.Button( leftFrame, text = "Pattern Computation", command = pattern_comp, width=20, height=4)
b2.pack( side="bottom")
b2.pack( padx= 250, pady = 45)
root.mainloop()

