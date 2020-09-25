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

generate_dca_matrix()

    ###############GUI#############


import tkinter as tk
from PIL import ImageTk, Image

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    # swap image to toggle
    def _img_swap(self):
        self.imgs[0], self.imgs[1] = self.imgs[1], self.imgs[0]

    def _toggle_image(self):
        img = self.imgs[1]
        self._img_swap()
        self.img_label.configure(image=img)

    def createCanvas(self, canvas_property):
        canvas = tk.Canvas(self)
        for k, v in canvas_property.items():
            canvas[k] = v
        print("argha")
        canvas.pack(side='top', fill=tk.X)
        canvas.create_image(950, 50, image=self.img)
        canvas.create_image(950, 200, image=self.img2)


    def createButton(self, btn_property):
        btn = tk.Button(self)
        for k, v in btn_property.items():
            btn[k] = v
        return btn

    def createImage(self, fig_path):
        img = ImageTk.PhotoImage(Image.open(fig_path))
        return img

    def createWidgets(self):
        self.img = self.createImage("protein1.png")
        self.img2 = self.createImage("protein2.png")
        self.img_label1 = tk.Label(self, image=self.img)
        self.img_label2 = tk.Label(self, image=self.img2)

        self.img_label1.pack(side=tk.TOP)
        self.img_label2.pack(side=tk.TOP)


        self.hi = self.createButton({
            'text': 'DCA Matrix',
            'width': '20',
            'height': '4',
            'command': self._toggle_image
        })

        self.QUIT = self.createButton({
            'text': 'Pattern Computation',
            'fg': 'red',
            'width': '20',
            'height':'4',
            'command': pattern_comp
        })

        # creating image
        img1 = self.createImage(fig_path='grey_1_pixel.png')
        imgo = self.createImage(fig_path='pattern_comp.png')

        self.img_label = tk.Label(self, image=img1)

        # image to use
        self.imgs = [img1, imgo]

        self.hi.pack(side=tk.TOP)
        self.QUIT.pack(side=tk.TOP)
        self.img_label.pack(side=tk.BOTTOM)


def create_gui():
    root = tk.Tk()
    label = tk.Label(root, text="filterDCA Input")
    label.pack(side="top")
    app = Application(master=root)
    app.mainloop()
    root.destroy()


def main():
    create_gui()


if __name__=='__main__':
    main()

