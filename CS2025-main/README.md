# 📘 Judul Proyek
*(Isi judul proyek Anda di sini)*

## 👤 Informasi
- **Nama:** Cipta Rangga Wijaya  
- **Repo:** [...]  
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan sesuai domain  
- Melakukan data preparation  
- Membangun 3 model: **Baseline**, **Advanced**, **Deep Learning**  
- Melakukan evaluasi dan menentukan model terbaik  

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Dataset sangat imbalanced (hanya 7.8% data kelas "Balanced")  
- Perlu model yang tetap akurat meski kelas minoritas sangat sedikit
- Membandingkan performa ML tradisional vs Deep Learning pada data tabular kecil

**Goals:**  
- Mencapai akurasi > 88% pada test set
- Menentukan model terbaik dari ketiga pendekatan

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_cnn.h5
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository  
- **Jumlah Data:** 625 instances   
- **Tipe:** Tabular (klasifikasi multi-kelas)   

### Fitur Utama
| Fitur             | Deskripsi                                   |
|-------------------|---------------------------------------------|
| Left_Weight       | Berat beban sisi kiri (1–5)                 |
| Left_Distance     | Jarak beban kiri dari tumpuan (1–5)         |
| Right_Weight      | Berat beban sisi kanan (1–5)                |
| Right_Distance    | Jarak beban kanan dari tumpuan (1–5)        |
| Class (target)    | L = Left tip, R = Right tip, B = Balanced   |
| Left_Moment*      | Left_Weight × Left_Distance (fitur baru)    |
| Right_Moment*     | Right_Weight × Right_Distance (fitur baru)  |

---

# 4. 🔧 Data Preparation
- Cleaning (missing/duplicate/outliers)  
- Transformasi (encoding/scaling)  
- Splitting (train/val/test)  

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Decision Tree Classifier → **87.20%**
- **Model 2 – Advanced ML:** Random Forest (500 trees, class_weight=balanced) → **92.80%**  
- **Model 3 – Deep Learning:** Multilayer Perceptron (128→64→32 neuron, 100+ epochs, EarlyStopping) → **91.20%**  

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy + F1-Score + Confusion Matrix

### Hasil Singkat
| Model              | Accuracy | F1-macro | Catatan                          |
|--------------------|----------|----------|----------------------------------|
| Baseline (DT)      | 87.20%   | 0.85     | Mudah overfit                    |
| Advanced (RF)      | **92.80%**   | **0.92** | **TERBAIK** – stabil & cepat    |
| Deep Learning (MLP)| 91.20%   | 0.90     | Butuh scaling & lebih lama       |

---

# 7. 🏁 Kesimpulan
- Model terbaik: Random Forest 
- Alasan: Akurasi tertinggi, F1-score terbaik, training cepat, interpretable  
- Insight penting: - Feature engineering sederhana (moment = weight × distance) sangat powerful  
  - Pada dataset tabular kecil & bersih, Random Forest > Deep Learning  
  - Deep Learning tidak selalu solusi terbaik

---

# 8. 🔮 Future Work
- [ ] Tambah data  
- [ ] Tuning model  
- [ ] Coba arsitektur DL lain  
- [ ] Deployment  

---

# 9. 🔁 Reproducibility
Gunakan environment:
```bash
pip install -r requirements.txt

# Jalankan notebook (recommended)
jupyter notebook notebooks/UAS_Balance_Scale_2025.ipynb

# Atau jalankan script sekali → semua model & gambar jadi
python src/train_models.py