import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

DATASET_DIR = "dataset"
OUTPUT_CSV = "dataset_fitur.csv"

data = []

labels = ["segar", "tidak_segar"]

for label in labels:
    folder_path = os.path.join(DATASET_DIR, label)

    if not os.path.exists(folder_path):
        print(f"Folder tidak ditemukan: {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Gagal membaca gambar: {image_path}")
            continue

        # Pre-processing
        img = cv2.resize(img, (224, 224))
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Fitur warna RGB
        mean_b = np.mean(img_blur[:, :, 0])
        mean_g = np.mean(img_blur[:, :, 1])
        mean_r = np.mean(img_blur[:, :, 2])

        # Fitur warna HSV
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv[:, :, 0])
        mean_s = np.mean(hsv[:, :, 1])
        mean_v = np.mean(hsv[:, :, 2])

        # Fitur tekstur GLCM
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

        data.append([
            filename,
            mean_r,
            mean_g,
            mean_b,
            mean_h,
            mean_s,
            mean_v,
            contrast,
            correlation,
            energy,
            homogeneity,
            label
        ])

df = pd.DataFrame(data, columns=[
    "filename",
    "R",
    "G",
    "B",
    "H",
    "S",
    "V",
    "contrast",
    "correlation",
    "energy",
    "homogeneity",
    "label"
])

df.to_csv(OUTPUT_CSV, index=False)

print("Ekstraksi fitur selesai.")
print("Jumlah data:", len(df))
print("File tersimpan:", OUTPUT_CSV)
print(df.head())