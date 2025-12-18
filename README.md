# Klasifikasi Keseimbangan Timbangan Menggunakan Machine Learning dan Deep Learning (Balance Scale Dataset)

## Informasi

- **Nama:** Cipta Rangga Wijaya
- **Repo:** https://github.com/CiptaRaW23/uas-datascience.git
- **Video:** https://youtu.be/TG_Geq3swlQ

---

# 1. Ringkasan Proyek

- Menyelesaikan permasalahan klasifikasi multi-kelas pada dataset Balance Scale (UCI)
- Melakukan data preparation lengkap termasuk feature engineering berbasis hukum fisika
- Membangun 3 model: **Baseline**, **Advanced**, **Deep Learning**
- Melakukan evaluasi dan menentukan model terbaik

---

# 2. Problem & Goals

**Problem Statements:**

- Dataset sangat imbalanced (hanya 7.8% data kelas "Balanced")
- Perlu model yang tetap akurat meski kelas minoritas sangat sedikit
- Membandingkan performa ML tradisional vs Deep Learning pada data tabular kecil

**Goals:**

- Mencapai akurasi > 88% pada test set
- Menentukan model terbaik dari ketiga pendekatan
- Menghasilkan proyek yang 100% reproducible dan sesuai standar PNM

---

## 📁 Struktur Folder

```
project/
│
├── data/
│   └── balance-scale.data          # Dataset (625 baris)
│
├── notebooks/
│   └── UAS_Balance_Scale_2025.ipynb
│
├── src/
│   └── train_models.py
│
├── models/
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_mlp.h5
│
├── images/
│   ├── class_distribution.png
│   ├── moment_scatter.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── mlp_history.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 3. 📊 Dataset

- **Sumber:** UCI Machine Learning Repository
- **Jumlah Data:** 625 instances
- **Tipe:** Tabular (klasifikasi multi-kelas)

### Fitur Utama

| Fitur          | Deskripsi                                  |
| -------------- | ------------------------------------------ |
| Left_Weight    | Berat beban sisi kiri (1–5)                |
| Left_Distance  | Jarak beban kiri dari tumpuan (1–5)        |
| Right_Weight   | Berat beban sisi kanan (1–5)               |
| Right_Distance | Jarak beban kanan dari tumpuan (1–5)       |
| Class (target) | L = Left tip, R = Right tip, B = Balanced  |
| Left_Moment\*  | Left_Weight × Left_Distance (fitur baru)   |
| Right_Moment\* | Right_Weight × Right_Distance (fitur baru) |

---

# 4. 🔧 Data Preparation

- **Cleaning**: Tidak diperlukan (tidak ada missing value, duplicate, atau outlier)
- **Transformasi**: Label encoding + StandardScaler (khusus untuk Deep Learning)
- **Feature Engineering**: Ditambahkan Left_Moment & Right_Moment
- **Splitting**: 80% train, 20% test (stratified, random_state=42)

---

# 5. 🤖 Modeling

- **Model 1 – Baseline:** Decision Tree Classifier → **100.00%**
- **Model 2 – Advanced ML:** Random Forest (500 trees, class_weight=balanced) → **87.20%**
- **Model 3 – Deep Learning:** Multilayer Perceptron (128→64→32 neuron, 100+ epochs, EarlyStopping) → **99.20%**

---

# 6. 🧪 Evaluation

**Metrik:** Accuracy + F1-Score + Confusion Matrix

### Hasil Singkat

| Model               | Accuracy    | F1-macro | Catatan                              |
| ------------------- | ----------- | -------- | ------------------------------------ |
| Baseline (DT)       | **100.00%** | 1.00     | **TERBAIK** – sempurna pada test set |
| Advanced (RF)       | 87.20%      | ~0.85    | Lebih stabil pada data baru          |
| Deep Learning (MLP) | 99.20%      | ~0.99    | Sangat baik tapi lebih kompleks      |

---

# 7. 🏁 Kesimpulan

- **Model terbaik:** Decision Tree
- **Alasan:** Akurasi sempurna (100.00%) pada test set, training tercepat, model paling sederhana dan interpretable
- **Insight penting:**
  - Feature engineering sederhana (moment = weight × distance) membuat data sangat mudah dipelajari
  - Pada dataset tabular kecil & bersih dengan pola fisika jelas, model baseline sederhana dapat mengungguli ensemble dan deep learning
  - Deep Learning memberikan hasil sangat baik tapi tidak selalu diperlukan

---

# 8. 🔮 Future Work

- [x] Tambah data
- [x] Tuning model
- [x] Coba arsitektur DL lain
- [ ] Deployment

---

# 9. 🔁 Reproducibility

Gunakan environment:

```bash
pip install -r requirements.txt

# Jalankan notebook (recommended)
jupyter notebook notebooks/UAS_Balance_Scale.ipynb

# Atau jalankan script sekali → semua model & gambar jadi
python src/train_models.py
```
