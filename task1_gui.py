
import tkinter 
from tkinter import ttk
from tkinter.constants import FLAT, LEFT, RIGHT, SOLID

root = tkinter.Tk()
root.title(u"EXP4-AUDIO-TASK1")
root.geometry("1000x600")


frame1 = tkinter.Frame(root, bd = 1, relief = SOLID, width = 600, height = 500)
frame1.pack(side=tkinter.LEFT, padx = 10)

frame2 = tkinter.Frame(root, width = 200, bd = 1, relief=SOLID, padx = 10, pady = 10, background="white")

label = tkinter.Label(frame2, text="グラフ選択", font=("Times New Roman", 24), anchor="w", background="white")

radiovalue = tkinter.IntVar()

bt_f0 = tkinter.Radiobutton(frame2, variable=radiovalue, value=1, text="基本周波数", font=("Times New Roman", 20), anchor="w", background="white")
bt_power = tkinter.Radiobutton(frame2, variable=radiovalue, value=2, text="音量", font=("Times New Roman", 20), anchor="w", background="white")

frame3 = tkinter.Frame(root, width = 200, bd = 1, relief = SOLID, padx = 10, pady = 10, background="white")

label2 = tkinter.Label(frame3, text="音声の再生", font=("Times New Roman", 24), anchor="w", background="white")

bt_pb = tkinter.Button(frame3, text="再生", font=("Times New Roman", 20),background="white")
bt_st = tkinter.Button(frame3, text="停止", font=("Times New Roman", 20),background="white")

frame2.pack(padx = 10, pady = 10)
label.pack()
bt_f0.pack()
bt_power.pack()

frame3.pack(padx = 10, pady = 10)
label2.pack()
bt_pb.pack(side = tkinter.LEFT)
bt_st.pack(side = tkinter.RIGHT, padx = 10)


root.mainloop()
