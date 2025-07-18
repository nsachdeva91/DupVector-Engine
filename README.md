<<<<<<< HEAD
# 🎬 Scene Matcher

A Python tool for detecting and deduplicating video scenes using perceptual hashing, MongoDB for metadata storage, and FAISS for fast similarity search. Includes a Streamlit dashboard for visual inspection.

---

## 📦 Setup Instructions

### 1. 🔧 Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

> 💡 On Windows, use `venv\Scripts\activate` instead of `source`.

---

### 2. 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. 🍃 Install MongoDB

#### macOS

```bash
brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb/brew/mongodb-community
```

#### Windows

1. Download from [MongoDB Community Server](https://www.mongodb.com/try/download/community)
2. Choose **MSI Installer** for your OS.
3. Use the **Complete Setup** option.
4. Enable MongoDB to run as a **Windows Service** during installation.

---

### 4. ✅ Start MongoDB

MongoDB runs as a service after installation.

#### Windows
- Open `services.msc`
- Look for **MongoDB Server**
- Ensure it's **Running**

---

## 🚀 Usage

### 1. Process Videos & Build FAISS Index

```bash
python3 main.py
```

- Scans the `videos/` folder
- Extracts scene features
- Stores metadata in MongoDB
- Builds FAISS vector index

### 2. Launch Streamlit Dashboard

```bash
streamlit run deduplicator.py
```

- Visual interface for checking duplicate/distinct scenes
- Allows selection and comparison of processed videos

---

## 📁 Folder Structure (Expected)

```
scene-matcher/
├── videos/               # Input video files
├── main.py               # Processing and indexing logic
├── deduplicator.py       # Streamlit dashboard
├── requirements.txt      # Python dependencies
└── README.md             # You're here
```

---

## 🧠 Notes

- Ensure MongoDB is running before executing any scripts.
- FAISS indices are stored locally and used for similarity queries.
- You can update the dashboard to include additional filters or visual features.

---

## 📃 License

MIT License
=======

# 🎬 DupVector Engine

This project is a multimodal video deduplication system that identifies **duplicate, variant, and distinct** video files using deep learning-based audio, video, and text similarity. Built with Streamlit for interactive dashboards, FAISS for vector search, and Transformer models for text embeddings.

---

## 🧠 Features

- 🔉 **Audio Fingerprinting** using MFCCs + cosine similarity
- 🎥 **Video Hashing** with perceptual image hashing (`pHash`)
- 📝 **Text Detection** from overlays using OCR + Transformer embeddings
- 🤖 **Similarity Search** with FAISS for fast vector comparisons
- 📊 **Interactive Dashboard** to visualize clusters, categories, and top matches
- 🎧 **Energy Plots** for audio comparison between files
- 🈶 **Language Detection** using Whisper for speech transcription
- 📂 **Duplicate Grouping** via Union-Find on pairwise similarity scores
- 🧮 **Clustering** via DBSCAN over weighted multimodal embeddings

---

## 🚀 Installation

git clone xxxxx
cd xxxxxx
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt


---

## 🗃️ Requirements

Ensure the following are installed:

Python ≥ 3.10

FFmpeg (for Whisper transcription and audio loading)

Tesseract-OCR (for embedded text extraction)

MongoDB (optional for metadata persistence)

---

## 🧪 Run the Deduplication Dashboard


streamlit run deduplicator.py

Upload a video file, or browse clustered duplicates from your indexed set.

---

## 📁 Directory Structure

<img width="355" height="104" alt="Image" src="https://github.com/user-attachments/assets/5c80fdfe-5b43-479d-ad39-683009868085" />

>>>>>>> 50a14779917f9825869422bf66cbe62672fdfc29
