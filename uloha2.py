import tkinter
import random
canvas=tkinter.Canvas(background="black",width=400, height=300)
canvas.pack()

basy = (31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)

def vykresli():
    canvas.delete("all")
    x = 20
    for p in basy:
        vyska = random.randint(20, 150)
        canvas.create_rectangle(x, 180 - vyska, x + 20, 180, fill="lime")
        canvas.create_text(x + 10, 190, text=str(p), fill="white", font=("Arial", 7))
        x += 35
    canvas.after(300, vykresli)

vykresli()

tkinter.mainloop()
