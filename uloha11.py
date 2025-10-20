import tkinter as tk

root = tk.Tk()
root.title("Hlavné okno")


def otvor_canvas2():
    okno2 = tk.Toplevel(root)
    okno2.title("Canvas 2")
    canvas2 = tk.Canvas(okno2, width=400, height=300, bg="lightyellow")
    canvas2.pack(padx=10, pady=10)
    canvas2.create_text(200, 150, text="Toto je Canvas 2", font=("Arial", 30, "bold"))

def otvor_canvas3():
    okno3 = tk.Toplevel(root)
    okno3.title("Canvas 3")
    canvas3 = tk.Canvas(okno3, width=400, height=300, bg="red")
    canvas3.pack(padx=10, pady=10)

   
    frame = tk.Frame(canvas3)
    listbox1 = tk.Listbox(frame)
    farby = ['red', 'green', 'blue', 'orange', "black", "yellow", "grey", "pink", "purple", "brown"]
    for prvok in farby:
        listbox1.insert('end', prvok)
    listbox1.pack()
    frame.pack()
    canvas3.create_window(200, 150, window=frame)

def otvor_canvas4():
    okno4 = tk.Toplevel(root)
    okno4.title("Canvas 4")
    canvas4 = tk.Canvas(okno4, width=400, height=300, bg="green")
    canvas4.pack(padx=10, pady=10)

    frame = tk.Frame(canvas4)
    checkbutton1 = tk.Checkbutton(frame, text='slovenský jazyk a literatúra')
    checkbutton2 = tk.Checkbutton(frame, text='matematika')
    checkbutton3 = tk.Checkbutton(frame, text='anglický jazyk')
    checkbutton4 = tk.Checkbutton(frame, text='informatika')
    checkbutton1.pack(anchor='w')
    checkbutton2.pack(anchor='w')
    checkbutton3.pack(anchor='w')
    checkbutton4.pack(anchor='w')

    canvas4.create_window(200, 150, window=frame)

def otvor_canvas5():
    okno5 = tk.Toplevel(root)
    okno5.title("Canvas 5")
    canvas5 = tk.Canvas(okno5, width=400, height=300, bg="blue")
    canvas5.pack(padx=10, pady=10)

    frame = tk.Frame(canvas5)
    radiobutton1 = tk.Radiobutton(frame, text='kruh', value=1)
    radiobutton2 = tk.Radiobutton(frame, text='štvorec', value=2)
    radiobutton3 = tk.Radiobutton(frame, text='trojuholník', value=3)
    radiobutton1.pack()
    radiobutton2.pack()
    radiobutton3.pack()

    canvas5.create_window(200, 150, window=frame)

def otvor_canvas6():
    okno6 = tk.Toplevel(root)
    okno6.title("Canvas 6")
    canvas6 = tk.Canvas(okno6, width=400, height=300, bg="orange")
    canvas6.pack(padx=10, pady=10)

    rx, ry = 100, 50
    x, y = 200, 150
    canvas6.create_oval(x - rx, y - ry, x + rx, y + ry, width=5, outline='green', tags='oval')

    def prekresli():
        canvas6.coords('oval', x - rx, y - ry, x + rx, y + ry)

    def zmena1(val):
        nonlocal rx
        rx = int(val)
        prekresli()

    def zmena2(val):
        nonlocal ry
        ry = int(val)
        prekresli()

    scale1 = tk.Scale(okno6, from_=10, to=200, orient='horizontal', length=400, command=zmena1)
    scale1.pack()
    scale1.set(rx)

    scale2 = tk.Scale(okno6, from_=10, to=150, orient='vertical', length=400, command=zmena2)
    scale2.place(x=380, y=20)
    scale2.set(ry)

def otvor_canvas7():
    okno7 = tk.Toplevel(root)
    okno7.title("Canvas 7")
    canvas7 = tk.Canvas(okno7, width=400, height=300, bg="yellow")
    canvas7.pack(padx=10, pady=10)

    frame = tk.Frame(canvas7)
    text1 = tk.Text(frame, height=10, width=30)
    text1.pack(side='left', fill='both', expand=True)
    scrollbar1 = tk.Scrollbar(frame, command=text1.yview)
    scrollbar1.pack(side='right', fill='y')
    text1.config(yscrollcommand=scrollbar1.set)

    canvas7.create_window(200, 150, window=frame)

    f = open('score.txt', 'r') 
    for riadok in f: 
        text1.insert('end', riadok) 
    f.close() 

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

zobrazit_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Zobraziť", menu=zobrazit_menu)

zobrazit_menu.add_command(label="Canvas 2", command=otvor_canvas2)
zobrazit_menu.add_command(label="Canvas 3", command=otvor_canvas3)
zobrazit_menu.add_command(label="Canvas 4", command=otvor_canvas4)
zobrazit_menu.add_command(label="Canvas 5", command=otvor_canvas5)
zobrazit_menu.add_command(label="Canvas 6", command=otvor_canvas6)
zobrazit_menu.add_command(label="Canvas 7", command=otvor_canvas7)


canvas1 = tk.Canvas(root, width=400, height=300, bg="white")
canvas1.pack(padx=10, pady=10)
canvas1.create_text(200, 150, text="Toto je Canvas 1", font=("Arial", 30, "bold"))

root.mainloop()
