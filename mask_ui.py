import tkinter as tk
from tkinter import Label
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk

model = load_model("mask_detector_model.h5")
categories = ['Mask', 'No Mask']

def detect_mask(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (100, 100)) / 255.0
        reshaped = np.reshape(resized, (1, 100, 100, 3))
        result = model.predict(reshaped)
        label = categories[np.argmax(result)]

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

def show_frame():
    _, frame = cap.read()
    frame = detect_mask(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

cap = cv2.VideoCapture(0)
root = tk.Tk()
root.title("Face Mask Detection")

lmain = Label(root)
lmain.pack()
show_frame()
root.mainloop()
