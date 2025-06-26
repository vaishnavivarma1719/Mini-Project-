import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the fine-tuned model
model = load_model("D:\\FINAL BRAIN\\best_finetuned_model.h5")

# Correct class names in alphabetical order of folder names used during training
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
pretty_names = {
    'glioma_tumor': 'Glioma Tumor',
    'meningioma_tumor': 'Meningioma Tumor',
    'no_tumor': 'No Tumor',
    'pituitary_tumor': 'Pituitary Tumor'
}

# Create main window
root = tk.Tk()
root.title("Brain Tumor Detector")
root.geometry("500x580")
root.configure(bg="#f0f4f8")

# Header label
tk.Label(
    root,
    text="Brain Tumor MRI Classifier",
    font=("Helvetica", 18, "bold"),
    bg="#f0f4f8",
    fg="#333"
).pack(pady=20)

# Image display area
img_label = tk.Label(root, bg="#f0f4f8")
img_label.pack(pady=10)

# Result text area
result_text = tk.StringVar()
tk.Label(
    root,
    textvariable=result_text,
    font=("Helvetica", 14),
    bg="#f0f4f8",
    fg="#007acc"
).pack(pady=10)

# Function to handle image selection and prediction
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            # Load and preprocess image
            img = Image.open(file_path).convert("RGB").resize((224, 224))
            tk_img = ImageTk.PhotoImage(img)
            img_label.config(image=tk_img)
            img_label.image = tk_img

            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array, verbose=0)
            class_index = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100
            predicted_label = pretty_names[class_names[class_index]]

            result_text.set(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%")

        except Exception as e:
            result_text.set(f"Failed to load/process image.\n{str(e)}")

# Button to select image
tk.Button(
    root,
    text="Select MRI Image",
    command=select_image,
    font=("Helvetica", 12),
    bg="#007acc",
    fg="white",
    padx=10,
    pady=5,
    relief="flat",
    cursor="hand2"
).pack(pady=20)

# Start the GUI loop
root.mainloop()
