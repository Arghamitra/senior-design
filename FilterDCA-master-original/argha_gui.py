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

    def createButton(self, btn_property):
        btn = tk.Button(self)
        for k, v in btn_property.items():
            btn[k] = v
        return btn

    def createImage(self, fig_path):
        img = ImageTk.PhotoImage(Image.open(fig_path))
        return img

    def createWidgets(self):
        self.QUIT = self.createButton({
            'text': 'Quit',
            'fg': 'red',
            'command': self.quit
        })

        self.hi = self.createButton({
            'text': 'Toggle img',
            'command': self._toggle_image
        })

        # creating image
        img1 = self.createImage(fig_path='figA.png')
        img2 = self.createImage(fig_path='figB.png')

        self.img_label = tk.Label(self, image=img1)

        # image to use
        self.imgs = [img1, img2]

        self.QUIT.pack(side=tk.TOP)
        self.hi.pack(side=tk.TOP)
        self.img_label.pack(side=tk.BOTTOM)


def create_gui():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
    root.destroy()


def main():
    create_gui()


if __name__=='__main__':
    main()

