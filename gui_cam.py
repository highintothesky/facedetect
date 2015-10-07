import Tkinter as tk
import cv2
from PIL import Image, ImageTk

# width, height = 800, 600
i = 0
cap = cv2.VideoCapture(0)
cap.set(3,384)
cap.set(4,286)

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()

l_detected = tk.Label(root)
l_detected.pack()

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    i = 8
    l_detected.configure(text=i)

show_frame()
root.mainloop()