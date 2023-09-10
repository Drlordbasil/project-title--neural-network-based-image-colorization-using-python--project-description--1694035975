import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox


class ImageColorizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = self.create_model()

    # Dataset Preparation
    def collect_dataset(self):
        dataset = []
        for file in os.listdir(self.dataset_path):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(self.dataset_path, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dataset.append(img)
        return np.array(dataset)

    def preprocess_dataset(self, dataset):
        grayscale_imgs = []
        color_imgs = []
        for img in dataset:
            grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            grayscale_dims = grayscale.reshape(
                (grayscale.shape[0], grayscale.shape[1], 1))
            grayscale_imgs.append(grayscale_dims)
            color_imgs.append(img)
        return np.array(grayscale_imgs), np.array(color_imgs)

    # Neural Network Architecture
    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(None, None, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2, activation='sigmoid'))
        return model

    # Model Training
    def train_model(self, X_train, y_train):
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=10,
                       batch_size=32, validation_split=0.2)

    def train(self):
        # Dataset Preparation
        dataset = self.collect_dataset()
        grayscale_imgs, color_imgs = self.preprocess_dataset(dataset)

        # Model Training
        self.train_model(grayscale_imgs, color_imgs)

    # Image Colorization
    def colorize_image(self, grayscale_img):
        grayscale_img = cv2.resize(grayscale_img, (256, 256))
        grayscale_img = grayscale_img.reshape(1, 256, 256, 1) / 255.0
        predicted_img = self.model.predict(grayscale_img)
        predicted_img = predicted_img.reshape(256, 256, 2) * 255.0
        colorized_img = np.concatenate((grayscale_img, predicted_img), axis=3)
        colorized_img = cv2.cvtColor(colorized_img[0], cv2.COLOR_YCrCb2RGB)
        return colorized_img


# User Interface
class UI:
    def __init__(self, image_colorizer):
        self.image_colorizer = image_colorizer

        self.window = tk.Tk()
        self.window.title("Image Colorization Tool")

        self.open_button = tk.Button(
            self.window, text="Open Image", command=self.open_image)
        self.open_button.pack()

        self.colorize_button = tk.Button(
            self.window, text="Colorize Image", command=self.colorize)
        self.colorize_button.pack()

        self.panel = tk.Label(self.window)
        self.panel.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename()
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

    def colorize(self):
        if self.image_colorizer.model is None:
            messagebox.showinfo(
                "Error", "Model not found. Please train the model first.")
            return
        file_path = filedialog.askopenfilename()
        grayscale_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        colorized_img = self.image_colorizer.colorize_image(grayscale_img)
        cv2.imshow("Colorized Image", colorized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def start(self):
        self.window.mainloop()


if __name__ == "__main__":
    dataset_path = "dataset"

    ic = ImageColorizer(dataset_path)
    ic.train()

    ui = UI(ic)
    ui.start()
