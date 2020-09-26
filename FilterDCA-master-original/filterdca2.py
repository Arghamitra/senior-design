#def createWidgets(self):

import tkinter as tk
from fdca_helper import *
import pandas
from PIL import ImageTk,Image

#def createWidgets():
root = tk.Tk()
topFrame = tk.Frame()
buttonFrame = tk.Frame()
bottomFrame = tk.Frame()

topFrame.pack(side="top", fill="both", expand=True)
buttonFrame.pack(side="top", fill="x")
bottomFrame.pack(side="bottom", fill="both", expand=True)

listBox = tk.Listbox(topFrame, width=30)
listBox.pack(side="top", fill="both", expand=True)

tk.Button(buttonFrame, text="Add").pack(side="left")
tk.Button(buttonFrame, text="Remove").pack(side="left")
tk.Button(buttonFrame, text="Edit").pack(side="left")

textBox = tk.Text(bottomFrame, height=10, width=30)
textBox.pack(fill="both", expand=True)

root.mainloop()