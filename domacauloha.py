import tkinter
import random
canvas=tkinter.Canvas()
canvas.pack()


pokusy = 0
cisla = random.randint(1, 1000)

def start():
    global cisla, pokusy
    pokusy += 1
    cisla = random.randint(1, 100)
    canvas.delete("all")
    canvas.create_text(190, 100, text="Myslíš si číslo " + str(cisla) + "? (pokus " + str(pokusy) + ")", font=("Arial", 16))

def mensie():
    start()

def vacsie():
    start()

def spravne():
    canvas.delete("all")
    canvas.create_text(150, 100, text="Uhádol som číslo " + str(cisla) + " na " + str(pokusy) + ". pokus!", font=("Arial", 13))


button1=tkinter.Button(text="Menšie", command=mensie)
button1.pack()
button2=tkinter.Button(text="Uhádol", command=spravne)
button2.pack()
button3=tkinter.Button(text="Väčšie", command=vacsie)
button3.pack()


start()
canvas.mainloop()
