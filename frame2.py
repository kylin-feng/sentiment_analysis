import tkinter as tk

from PIL import ImageTk,Image

import os
from tkinter import font
# 创建窗口
window = tk.Tk()
window.title("情绪分析系统")
window.geometry("541x319")

image = Image.open("background3.png")
photo = ImageTk.PhotoImage(image)
bg_label = tk.Label(window, image=photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
# 创建文本框

def analyze_text():
    os.system('python real_time_video.py')

# 定义情绪分析函数
# 创建按钮
button_font = font.Font(weight="bold")
button = tk.Button(window, text="完成", command=analyze_text,width=8,height = 3,font = button_font)
button.pack()
button.place(x=155,y=135)
button.config(bg="beige",fg="red")


# 进入消息循环
window.mainloop()