# 🐟 Klasifikasi Kesegaran Ikan Tongkol Berdasarkan Citra Mata Menggunakan Random Forest

## 📖 Deskripsi Proyek

Proyek ini merupakan penelitian di bidang Pengolahan Citra Digital (PCD) yang bertujuan untuk mengklasifikasikan tingkat kesegaran ikan tongkol berdasarkan citra mata ikan menggunakan algoritma **Random Forest**.

Mata ikan dipilih sebagai objek penelitian karena merupakan salah satu indikator visual yang dapat menunjukkan tingkat kesegaran ikan. Perubahan warna, kejernihan, dan tekstur pada mata ikan dapat digunakan untuk membedakan kondisi ikan yang masih segar maupun yang sudah tidak segar.

Dataset yang digunakan dalam penelitian ini merupakan **181 citra primer** yang diperoleh secara langsung melalui pengambilan gambar di lapangan. Setiap citra kemudian diberi label ke dalam dua kelas, yaitu:

- Segar
- Tidak Segar

---

## 🔄 Metodologi Penelitian

### 1. Pengumpulan Data
Data diperoleh melalui pengambilan citra mata ikan tongkol secara langsung di lapangan menggunakan kamera smartphone.

Jumlah dataset:

- Total data: **181 citra**
- Kelas Segar: **115 citra**
- Kelas Tidak Segar: **66 citra**

---

### 2. Pelabelan Data
Setiap citra diberi label berdasarkan kondisi visual mata ikan menjadi:

- **Segar**
- **Tidak Segar**

---

### 3. Pre-processing Citra

Sebelum dilakukan ekstraksi fitur, setiap citra melalui beberapa tahapan pre-processing:

- Cropping area mata ikan
- Resize menjadi **224 × 224 piksel**
- Gaussian Blur untuk reduksi noise
- Konversi warna RGB ke HSV
- Konversi RGB ke Grayscale

---

### 4. Ekstraksi Fitur

Penelitian ini menggunakan kombinasi fitur warna dan fitur tekstur.

#### Fitur Warna (6 Fitur)

- Mean Red (R)
- Mean Green (G)
- Mean Blue (B)
- Mean Hue (H)
- Mean Saturation (S)
- Mean Value (V)

#### Fitur Tekstur GLCM (4 Fitur)

- Contrast
- Correlation
- Energy
- Homogeneity

Total fitur yang digunakan:

**10 fitur numerik**

---

### 5. Pembentukan Dataset Fitur

Seluruh fitur hasil ekstraksi digabungkan ke dalam dataset numerik yang digunakan sebagai input algoritma machine learning.

Jumlah data akhir:

- 181 data
- 10 fitur
- 2 kelas

---

### 6. Pembagian Data

Dataset dibagi menggunakan metode hold-out:

- Training Data : 80% (144 data)
- Testing Data : 20% (37 data)

```python
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

### 7. Klasifikasi Menggunakan Random Forest

Model klasifikasi dibangun menggunakan algoritma **Random Forest**.

Parameter yang digunakan:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None
)
```

Tahapan Random Forest:

1. Bootstrap Sampling
2. Pemilihan fitur secara acak
3. Pembentukan banyak Decision Tree
4. Prediksi setiap tree
5. Majority Voting
6. Penentuan hasil klasifikasi akhir

---

### 8. Evaluasi Model

Performa model dievaluasi menggunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Hasil terbaik yang diperoleh:

| Metrik | Nilai |
|---------|---------|
| Accuracy | **89%** |
| Precision | 0.89 |
| Recall | 0.86 |
| F1-Score | 0.88 |

Confusion Matrix:

```text
[[23  1]
 [ 3 10]]
```

---

## 🖥️ Teknologi yang Digunakan

- Python
- OpenCV
- Scikit-Learn
- Scikit-Image
- NumPy
- Pandas
- Joblib
- Tkinter

---

## 📂 Struktur Folder

```text
model/
│
├── dataset/
│   ├── segar/
│   └── tidak_segar/
│
├── ekstraksi_fitur.py
├── train_random_forest.py
├── app_prediksi.py
├── dataset_fitur.csv
├── model_random_forest_ikan.pkl
└── requirements.txt
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependency

```bash
pip install -r requirements.txt
```

### 2. Ekstraksi Fitur

```bash
python ekstraksi_fitur.py
```

### 3. Training Model

```bash
python train_random_forest.py
```

### 4. Jalankan Aplikasi Prediksi

```bash
python app_prediksi.py
```

---

## 🎯 Hasil Akhir

Sistem mampu melakukan klasifikasi kesegaran ikan tongkol berdasarkan citra mata ikan dan menghasilkan output:

✅ **Segar**

atau

❌ **Tidak Segar**

beserta tingkat keyakinan (confidence score) yang dihasilkan oleh model Random Forest.

---

## 👨‍💻 Tim Pengembang

Proyek ini dikembangkan sebagai tugas penelitian pada mata kuliah Pengolahan Citra Digital (PCD) dengan fokus pada penerapan Machine Learning untuk identifikasi kesegaran ikan tongkol berbasis citra mata.
