# 第二个控件，Button
import tkinter

top = tkinter.Tk()
# 在 Mac 的黑暗模式中，会显示不出来文字，所以，要用 Light
btn = tkinter.Button(top, command=top.quit, text="abaozai")
btn.pack()
tkinter.mainloop()

