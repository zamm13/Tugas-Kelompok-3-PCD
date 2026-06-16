import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

DATASET_PATH = "dataset_fitur.csv"
MODEL_PATH = "model_random_forest_ikan.pkl"


def jalankan_training():
    try:
        df = pd.read_csv(DATASET_PATH)

        X = df.drop(["filename", "label"], axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        labels = sorted(y.unique())

        cm = confusion_matrix(y_test, y_pred, labels=labels)

        report = classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=[label.replace("_", " ").upper() for label in labels]
        )

        joblib.dump(model, MODEL_PATH)

        tampilkan_hasil(
            df=df,
            labels=labels,
            cm=cm,
            report=report,
            accuracy=accuracy,
            train_count=len(X_train),
            test_count=len(X_test)
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))


def tampilkan_hasil(df, labels, cm, report, accuracy, train_count, test_count):
    root = tk.Tk()
    root.title("Hasil Evaluasi Model Random Forest")
    root.geometry("760x680")
    root.configure(bg="#f4f6f8")
    root.resizable(False, False)

    title = tk.Label(
        root,
        text="Evaluasi Model Random Forest",
        font=("Segoe UI", 20, "bold"),
        bg="#f4f6f8",
        fg="#111827"
    )
    title.pack(pady=(20, 5))

    subtitle = tk.Label(
        root,
        text="Klasifikasi Kesegaran Ikan Tongkol Berdasarkan Citra Mata",
        font=("Segoe UI", 10),
        bg="#f4f6f8",
        fg="#6b7280"
    )
    subtitle.pack(pady=(0, 15))

    summary_frame = tk.Frame(root, bg="#f4f6f8")
    summary_frame.pack(pady=10)

    def buat_card(parent, judul, nilai, warna):
        card = tk.Frame(
            parent,
            bg="white",
            width=160,
            height=85,
            highlightbackground="#d1d5db",
            highlightthickness=1
        )
        card.pack(side="left", padx=8)
        card.pack_propagate(False)

        lbl_judul = tk.Label(
            card,
            text=judul,
            font=("Segoe UI", 9),
            bg="white",
            fg="#6b7280"
        )
        lbl_judul.pack(pady=(12, 2))

        lbl_nilai = tk.Label(
            card,
            text=nilai,
            font=("Segoe UI", 17, "bold"),
            bg="white",
            fg=warna
        )
        lbl_nilai.pack()

    buat_card(summary_frame, "Total Dataset", str(len(df)), "#111827")
    buat_card(summary_frame, "Data Training", str(train_count), "#2563eb")
    buat_card(summary_frame, "Data Testing", str(test_count), "#7c3aed")
    buat_card(summary_frame, "Akurasi", f"{accuracy * 100:.2f}%", "#16a34a")

    label_frame = tk.LabelFrame(
        root,
        text="Distribusi Label Dataset",
        font=("Segoe UI", 10, "bold"),
        bg="#f4f6f8",
        fg="#111827",
        padx=10,
        pady=10
    )
    label_frame.pack(fill="x", padx=35, pady=10)

    distribusi = df["label"].value_counts()

    for label, jumlah in distribusi.items():
        teks = f"{label.replace('_', ' ').upper()} : {jumlah} data"
        tk.Label(
            label_frame,
            text=teks,
            font=("Segoe UI", 10),
            bg="#f4f6f8",
            fg="#374151"
        ).pack(anchor="w")

    cm_frame = tk.LabelFrame(
        root,
        text="Confusion Matrix",
        font=("Segoe UI", 10, "bold"),
        bg="#f4f6f8",
        fg="#111827",
        padx=10,
        pady=10
    )
    cm_frame.pack(fill="x", padx=35, pady=10)

    columns = ["Aktual / Prediksi"] + [
        label.replace("_", " ").upper() for label in labels
    ]

    tree = ttk.Treeview(
        cm_frame,
        columns=columns,
        show="headings",
        height=len(labels)
    )

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=160)

    for i, label in enumerate(labels):
        row = [label.replace("_", " ").upper()] + list(cm[i])
        tree.insert("", "end", values=row)

    tree.pack(fill="x")

    report_frame = tk.LabelFrame(
        root,
        text="Classification Report",
        font=("Segoe UI", 10, "bold"),
        bg="#f4f6f8",
        fg="#111827",
        padx=10,
        pady=10
    )
    report_frame.pack(fill="both", expand=True, padx=35, pady=10)

    report_text = scrolledtext.ScrolledText(
        report_frame,
        font=("Consolas", 10),
        bg="white",
        fg="#111827",
        height=10,
        relief="flat"
    )
    report_text.pack(fill="both", expand=True)

    hasil_laporan = f"""
HASIL EVALUASI MODEL RANDOM FOREST

Dataset        : {len(df)} data primer
Data Training  : {train_count} data
Data Testing   : {test_count} data
Akurasi Model  : {accuracy * 100:.2f}%

Classification Report:
{report}

Model berhasil disimpan sebagai:
{MODEL_PATH}
"""

    report_text.insert(tk.END, hasil_laporan)
    report_text.config(state="disabled")

    footer = tk.Label(
        root,
        text="Dataset: 181 data primer | Metode: Random Forest | Citra Mata Ikan Tongkol",
        font=("Segoe UI", 9),
        bg="#f4f6f8",
        fg="#6b7280"
    )
    footer.pack(pady=(0, 12))

    root.mainloop()


jalankan_training()