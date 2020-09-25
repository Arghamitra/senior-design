import tkinter as tk
from PIL import Image, ImageTk


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(self, master)
        self.number = 0
        self.widgets = []
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        self.cloneButton = tk.Button(self, text='Clone', command=self.clone)
        self.cloneButton.grid()

    def clone(self):
        widget = tk.Label(self, text='label #%s' % self.number)
        widget.grid()
        self.widgets.append(widget)
        self.number += 1

    def switch_img(self, img_path='protein1.png'):
        self.img = tk.PhotoImage(Image.open(img_path))
        self.pack()


if __name__ == "__main__":
    app = Application()
    app.master.title("Sample application")
    app.mainloop()