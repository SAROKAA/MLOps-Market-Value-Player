# ⚽ MLOps: Prediksi Harga Pemain Sepak Bola

Proyek ini merupakan implementasi **pipeline MLOps** untuk memprediksi nilai pasar (_market value_) pemain sepak bola berdasarkan performa statistik historis. Sistem ini mengintegrasikan proses pengambilan data otomatis, _feature engineering_, serta **Data Version Control (DVC)** untuk mendukung pengembangan model secara berkelanjutan (_continual learning_).

---

## 🎯 Tujuan Proyek

- Membangun pipeline data otomatis dari berbagai endpoint API (statistik & transfer)
- Melakukan **data fusion** antara performa teknis dan nilai ekonomi pemain
- Mengimplementasikan **DVC** untuk melacak versi dataset tanpa membebani Git
- Mendukung _continuous training_ dengan dataset yang terstruktur dan terversi

---

## 📂 Struktur Direktori

```
MLOps-Prediksi-Harga-Pemain-Sepak-Bola/
├── data/
│   ├── raw/                         # Data mentah (JSON)
│   │   ├── league_621/              # Statistik pemain per tim
│   │   └── transfers/               # Riwayat transfer pemain
│   └── processed/                   # Data hasil olahan (CSV)
│       └── data_processed_YYYYMMDD_HHMMSS.csv
├── src/
│   └── data/                        # Modul Data Engineering
│       ├── ingest_data.py           # Ingestion statistik pemain
│       ├── ingest_transfers_per_team.py  # Ingestion data transfer
│       └── preprocess.py            # Preprocessing & feature engineering
├── models/                          # Model machine learning
├── .dvc/                            # Konfigurasi internal DVC
├── .env                             # API Key (jangan di-commit)
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Dependency Python
└── README.md                        # Dokumentasi proyek
```

---

## 🛠️ Panduan Menjalankan Pipeline

### 1️⃣ Setup Environment

```bash
# Install dependency
pip install -r requirements.txt

# Setup .env dengan API Key
# Tambahkan: RAPIDAPI_KEY=your_api_key_here
```

### 2️⃣ Ingest Data (Data Acquisition)

```bash
# Ambil statistik pemain dari liga
python src/data/ingest_data.py

# Ambil data transfer pemain per tim
# (Sesuaikan variabel target_team di dalam script)
python src/data/ingest_transfers_per_team.py
```

### 3️⃣ Preprocessing & Feature Engineering

Menggabungkan data statistik dengan transfer history, membersihkan format, dan menghasilkan fitur tambahan:

```bash
python src/data/preprocess.py
```

**Output:** `data/processed/data_processed_YYYYMMDD_HHMMSS.csv`

---

## 📊 Data Versioning dengan DVC

### 🚀 Inisialisasi Tracking Dataset

```bash
# Inisialisasi DVC
dvc init

# Track dataset terprocessing
dvc add data/processed/data_processed_*.csv

# Simpan metadata ke Git
git add data/processed/data_processed_*.csv.dvc data/processed/.gitignore
git commit -m "Track processed dataset (v1)"
```

### 🔄 Update Dataset (Continual Learning)

Ketika ada data baru atau perubahan:

```bash
# Jalankan ulang pipeline
python src/data/preprocess.py

# Cek perubahan dataset
dvc status
dvc diff

# Update tracking
dvc add data/processed/data_processed_*.csv
git add data/processed/data_processed_*.csv.dvc
git commit -m "Update dataset (v2)"
```

---

## 📈 Metodologi Sains Data

Pipeline ini mengikuti tahapan berikut:

### 🔍 Data Discovery

- Identifikasi `player_id` untuk menghubungkan data statistik dan transfer

### 🧹 Data Preparation

- Konversi format data (string → numerik)
- Membersihkan simbol mata uang (€ 74M → 74000000.0)
- Menghapus data tidak relevan (noise)

### 🔗 Data Fusion

- Menggabungkan fitur performa dengan nilai pasar (market value)
- Feature engineering: goal contribution, shot accuracy, pass ratio, dll

### 📝 Feature List

**Basic Stats:**

- `goals`, `assists`, `goal_contribution`
- `rating`, `minutes`, `appearances`, `position`

**Advanced Stats:**

- **Shooting:** `shots_total`, `shots_on`, `shot_accuracy`
- **Passing:** `passes_total`, `passes_key`, `pass_ratio`
- **Defense:** `tackles_total`, `tackles_blocks`, `tackles_interceptions`, `defensive_actions`
- **Dribbling:** `dribbles_success`, `dribbles_attempts`, `dribble_success_rate`
- **Duels:** `duels_won`, `duels_total`, `duel_win_rate`
- **Discipline:** `cards_yellow`, `cards_red`, `discipline_score`
- **Keeper:** `saves`, `conceded` (untuk goalkeeper)
- **Other:** `fouls_committed`, `fouls_drawn`, `penalty_scored`, `penalty_missed`

### 🗂️ Data Versioning

- Menggunakan DVC untuk menjaga integritas dan traceability dataset
- Hash (MD5) menjamin reproducibility

---

## 💻 Menjalankan di GitHub Codespaces

1. Buka repository di GitHub
2. Klik: **Code** → **Codespaces** → **Create Codespace**
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan pipeline sesuai panduan di atas

---
