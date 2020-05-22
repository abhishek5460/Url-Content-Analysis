from tkinter import *
window=Tk()
lbl=Entry(window)
lbl.place(x=60, y=50)
window.title('Hello Python')
window.geometry("300x200+10+10")
print(lbl.get())
window.mainloop()
