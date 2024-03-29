import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf  # Replace with your actual model import

# Create the main application window
app = tk.Tk()
app.title("Malaria Detector")

# Load your trained machine learning model here
# Replace 'your_model_path' with the actual path to your model file
model = tf.keras.models.load_model('malaria_detection_model.h5')

# Global variables to store the selected image and classification result
selected_image = None
classification_result = tk.StringVar()

# Function to upload and display an image
def upload_image():
    global selected_image, classification_result
    file_path = filedialog.askopenfilename()  # Open a file dialog to choose an image
    if file_path:
        # Clear the previous classification result
        classification_result.set("")

        # Display the selected image in the GUI
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        selected_image = file_path

# Function to classify the uploaded image
def classify_image():
    global selected_image, classification_result
    if selected_image:
        # Load and preprocess the image for inference (you'll need to adapt this part)
        image = Image.open(selected_image)
        image = image.resize((64, 64))  # Resize the image to match your model's input size
        image = np.array(image) / 255.0  # Normalize the image data
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Perform inference with your model
        prediction = model.predict(image)

        # Determine the class (Infected or Uninfected)
        if prediction[0][0] > 0.5:
            classification_result.set("Result: Infected")
        else:
            classification_result.set("Result: Uninfected")
    else:
        classification_result.set("Please upload an image first.")

# Create a frame for grouping widgets and adding padding
frame = tk.Frame(app, bg="white", padx=20, pady=20)
frame.pack(expand=True, fill=tk.BOTH)

# Create labels, buttons, and image display area with decorations
title_label = tk.Label(
    frame,
    text="Malaria Detector",
    font=("Helvetica", 20, "bold"),
    bg="blue",
    fg="white"
)
title_label.pack(fill=tk.X, pady=(0, 10))

upload_button = tk.Button(
    frame,
    text="Upload Image",
    command=upload_image,
    font=("Helvetica", 14),
    bg="orange",
    fg="white",
    padx=10,
    pady=5
)
upload_button.pack(pady=10)

image_label = tk.Label(frame)
image_label.pack()

classify_button = tk.Button(
    frame,
    text="Classify Image",
    command=classify_image,
    font=("Helvetica", 14),
    bg="green",
    fg="white",
    padx=10,
    pady=5
)
classify_button.pack(pady=10)

result_label = tk.Label(
    frame,
    textvariable=classification_result,
    font=("Helvetica", 16),
    fg="black"
)
result_label.pack(pady=10)

# Start the Tkinter main loop
app.mainloop()