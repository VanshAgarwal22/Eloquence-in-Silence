import tkinter as tk
from tkinter import ttk
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import Image, ImageTk
import numpy as np
import math

class HandDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Detection")
        self.root.geometry("1000x600")
        self.root.configure(bg="#1abc9c")

        # Initialize styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#16a085')
        self.style.configure('TLabel', background='#16a085', font=('Helvetica', 16), foreground='white')
        self.style.configure('Title.TLabel', font=('Helvetica', 24, 'bold'), foreground='#ecf0f1')
        self.style.configure('TButton', font=('Helvetica', 12), foreground='black')

        # Setup GUI components
        self.setup_gui()

        # Initialize video capture and hand detector
        self.cap = None
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier("C:\\Users\\Vansh\\Desktop\\sl\\model\\keras_model.h5", 
                                     "C:\\Users\\Vansh\\Desktop\\sl\\model\\labels.txt")
        self.offset = 20
        self.imgSize = 300
        self.labels = ["Hello", "I love you", "Ok", "Yes", "Thank You", "Promise", "Very Good", "A", "B", "C", "D", "Very Bad"]

        # Hand connections for drawing hand landmarks
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
        ]

        self.video_running = False

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Video frame
        video_frame = ttk.Frame(main_frame, padding="10 10 10 10")
        video_frame.grid(row=0, column=0, sticky=tk.N)
        ttk.Label(video_frame, text="Video Feed", style='Title.TLabel').pack()

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(padx=10, pady=10)

        # Text frame
        text_frame = ttk.Frame(main_frame, padding="10 10 10 10")
        text_frame.grid(row=1, column=0, sticky=tk.N)
        ttk.Label(text_frame, text="Detected Gesture", style='Title.TLabel').pack()

        self.text_label = ttk.Label(text_frame, text="")
        self.text_label.pack(padx=10, pady=10)

        # Drawing frame
        drawing_frame = ttk.Frame(main_frame, padding="10 10 10 10")
        drawing_frame.grid(row=0, column=1, rowspan=2, sticky=tk.N)
        ttk.Label(drawing_frame, text="Hand Drawing", style='Title.TLabel').pack()

        self.drawing_label = ttk.Label(drawing_frame)
        self.drawing_label.pack(padx=10, pady=10)

        # Control frame
        control_frame = ttk.Frame(main_frame, padding="10 10 10 10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW)
        self.start_button = ttk.Button(control_frame, text="Start Video", command=self.start_video)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = ttk.Button(control_frame, text="Stop Video", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
    def start_video(self):
        if not self.video_running:
            self.cap = cv2.VideoCapture(0)
            self.video_running = True
            self.update_frame()

    def stop_video(self):
        if self.video_running:
            self.video_running = False
            self.cap.release()
            self.video_label.config(image='')
            self.drawing_label.config(image='')
            self.text_label.config(text='')

    def preprocess_image(self, img, hand_bbox):
        x, y, w, h = hand_bbox
        imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
        imgCrop = img[y-self.offset:y + h + self.offset, x-self.offset:x + w + self.offset]
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                wGap = math.ceil((self.imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = self.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                hGap = math.ceil((self.imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
        except Exception as e:
            print("Error during image preprocessing:", e)
            return None

        return imgWhite

    def update_frame(self):
        if self.video_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                imgOutput = frame.copy()
                hands, img = self.detector.findHands(frame_rgb)

                gesture_text = ""
                white_image = np.ones((480, 640, 3), dtype=np.uint8) * 255

                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    imgWhite = self.preprocess_image(img, (x, y, w, h))

                    if imgWhite is not None:
                        prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                        if 0 <= index < len(self.labels):
                            gesture_text = self.labels[index]
                        else:
                            gesture_text = "Unknown"

                        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 70), 
                                      (x - self.offset + 400, y - self.offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                        cv2.putText(imgOutput, gesture_text, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                        cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), 
                                      (x + w + self.offset, y + h + self.offset), (0, 255, 0), 4)

                        white_image = self.draw_hand_on_white_image(hand)

                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                draw_img = Image.fromarray(white_image)
                draw_imgtk = ImageTk.PhotoImage(image=draw_img)
                self.drawing_label.imgtk = draw_imgtk
                self.drawing_label.configure(image=draw_imgtk)

                self.text_label.config(text=gesture_text)

            self.root.after(10, self.update_frame)

    def draw_hand_on_white_image(self, hand):
        white_image = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Create a white image
        hand_landmarks = hand["lmList"]
        for lm in hand_landmarks:
            cx, cy, _ = lm  # Only use x and y coordinates
            cv2.circle(white_image, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
        
        for connection in self.hand_connections:
            start_point = hand_landmarks[connection[0]]
            end_point = hand_landmarks[connection[1]]
            cv2.line(white_image, (start_point[0], start_point[1]), (end_point[0], end_point[1]), (0, 0, 0), 2)
        
        return white_image

    def on_closing(self):
        if self.video_running:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
