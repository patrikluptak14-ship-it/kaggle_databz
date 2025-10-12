import tkinter
import random
canvas=tkinter.Canvas(width=350, height=150)
canvas.pack()



zapalky = 15
hrac = 1

def zobraz():
    canvas.delete("all")
    x = 20
    for i in range(zapalky):
        canvas.create_line(x, 50, x, 150, width=5, fill="orange")
        canvas.create_oval(x-5, 40, x+5, 50, fill="red")
        x += 20
    canvas.create_text(150, 180, text="Hráč " + str(hrac) + " je na ťahu", font=("Arial", 14))

def vezmi1():
    vezmi(1)

def vezmi2():
    vezmi(2)

def vezmi3():
    vezmi(3)

def vezmi(n):
    global zapalky, hrac
    if zapalky > 0:
        zapalky -= n
        if zapalky <= 0:
            canvas.delete("all")
            canvas.create_text(150, 100, text="Hráč " + str(hrac) + " prehral!", font=("Arial", 16))
        else:
            if hrac == 1:
                hrac = 2
            else:
                hrac = 1
            zobraz()


tlacidlo1 = tkinter.Button(text="Vezmi 1", command=vezmi1)
tlacidlo1.pack()

tlacidlo2 = tkinter.Button(text="Vezmi 2", command=vezmi2)
tlacidlo2.pack()

tlacidlo3 = tkinter.Button(text="Vezmi 3", command=vezmi3)
tlacidlo3.pack()

zobraz()

tkinter.mainloop()
