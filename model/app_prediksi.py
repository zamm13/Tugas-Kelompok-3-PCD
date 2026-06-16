import cv2
import joblib
import numpy as np
import tkinter as tk
import pandas as pd
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops

MODEL_PATH = "model_random_forest_ikan.pkl"

FEATURE_COLUMNS = [
    "R",
    "G",
    "B",
    "H",
    "S",
    "V",
    "contrast",
    "correlation",
    "energy",
    "homogeneity"
]

model = joblib.load(MODEL_PATH)


def ekstraksi_fitur(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Gambar tidak dapat dibaca.")

    # Pre-processing
    img = cv2.resize(img, (224, 224))
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Ekstraksi fitur RGB
    mean_b = np.mean(img_blur[:, :, 0])
    mean_g = np.mean(img_blur[:, :, 1])
    mean_r = np.mean(img_blur[:, :, 2])

    # Ekstraksi fitur HSV
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])

    # Ekstraksi fitur tekstur GLCM
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    fitur = pd.DataFrame([[
        mean_r,
        mean_g,
        mean_b,
        mean_h,
        mean_s,
        mean_v,
        contrast,
        correlation,
        energy,
        homogeneity
    ]], columns=FEATURE_COLUMNS)

    return fitur


def pilih_gambar():
    file_path = filedialog.askopenfilename(
        title="Pilih Citra Mata Ikan",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img = img.resize((260, 260))
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk, text="")
        image_label.image = img_tk

        fitur = ekstraksi_fitur(file_path)
        prediksi = model.predict(fitur)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(fitur)[0]
            confidence = max(proba) * 100
        else:
            confidence = 0

        hasil_text = str(prediksi).replace("_", " ").upper()

        result_label.config(
            text=f"Hasil Prediksi: {hasil_text}"
        )

        confidence_label.config(
            text=f"Tingkat Keyakinan: {confidence:.2f}%"
        )

        if prediksi == "segar":
            status_label.config(
                text="Ikan teridentifikasi dalam kondisi SEGAR",
                fg="#16a34a"
            )
        else:
            status_label.config(
                text="Ikan teridentifikasi dalam kondisi TIDAK SEGAR",
                fg="#dc2626"
            )

    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Klasifikasi Kesegaran Ikan Tongkol")
root.geometry("520x650")
root.configure(bg="#f4f6f8")
root.resizable(False, False)

title = tk.Label(
    root,
    text="Klasifikasi Kesegaran Ikan Tongkol",
    font=("Segoe UI", 18, "bold"),
    bg="#f4f6f8",
    fg="#111827"
)
title.pack(pady=(25, 5))

subtitle = tk.Label(
    root,
    text="Berdasarkan Citra Mata Menggunakan Random Forest",
    font=("Segoe UI", 10),
    bg="#f4f6f8",
    fg="#6b7280"
)
subtitle.pack(pady=(0, 20))

card = tk.Frame(
    root,
    bg="white",
    width=420,
    height=420,
    highlightbackground="#d1d5db",
    highlightthickness=1
)
card.pack(pady=10)
card.pack_propagate(False)

image_label = tk.Label(
    card,
    text="Belum ada gambar dipilih",
    font=("Segoe UI", 11),
    bg="white",
    fg="#9ca3af"
)
image_label.pack(expand=True)

button = tk.Button(
    root,
    text="Pilih Gambar Mata Ikan",
    font=("Segoe UI", 11, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    relief="flat",
    padx=25,
    pady=10,
    cursor="hand2",
    command=pilih_gambar
)
button.pack(pady=20)

result_label = tk.Label(
    root,
    text="Hasil Prediksi: -",
    font=("Segoe UI", 14, "bold"),
    bg="#f4f6f8",
    fg="#111827"
)
result_label.pack(pady=(10, 5))

confidence_label = tk.Label(
    root,
    text="Tingkat Keyakinan: -",
    font=("Segoe UI", 11),
    bg="#f4f6f8",
    fg="#374151"
)
confidence_label.pack(pady=5)

status_label = tk.Label(
    root,
    text="Silakan pilih citra mata ikan untuk diprediksi",
    font=("Segoe UI", 10),
    bg="#f4f6f8",
    fg="#6b7280"
)
status_label.pack(pady=10)

footer = tk.Label(
    root,
    text="Dataset: 181 data primer | Akurasi Model: 89%",
    font=("Segoe UI", 9),
    bg="#f4f6f8",
    fg="#6b7280"
)
footer.pack(side="bottom", pady=15)

root.mainloop()
