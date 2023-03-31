import tkinter as tk

from PIL import ImageTk,Image

import os
from tkinter import font
# 创建窗口
window = tk.Tk()
window.title("情绪分析系统")
window.geometry("697x529")

image = Image.open("background2.png")
photo = ImageTk.PhotoImage(image)
bg_label = tk.Label(window, image=photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
# 创建文本框

def analyze_text():
    os.system('python frame1.py')

# 定义情绪分析函数
# 创建按钮
button_font = font.Font(weight="bold")
button = tk.Button(window, text="开始分析", command=analyze_text,width=8,height = 3,font = button_font)
button.pack()
button.place(x=510,y=350)
button.config(bg="beige",fg="red")


# 进入消息循环
window.mainloop()