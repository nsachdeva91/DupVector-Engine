# video_processor.py
import os, time, cv2, pickle
import numpy as np
from PIL import Image
import imagehash
from datetime import datetime
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector
from pymongo import MongoClient
import faiss
from pydub import AudioSegment
from python_speech_features import mfcc
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import pytesseract
import re

# === CONFIGURATION ===
SAMPLE_RATE = 16000
META_PATH = "metadata.pkl"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class VideoProcessor:
    def __init__(self, db, video_path, output_dir, downscale=(256, 256), skip_frames=2, threshold=30.0):
        self.db = db
        self.video_path = video_path
        self.downscale = downscale
        self.skip_frames = skip_frames
        self.threshold = threshold
        self.output_dir = output_dir

    def _detect_scenes(self):
        video = open_video(self.video_path)
        video.skip_frames = self.skip_frames
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=self.threshold))
        video.seek(0)
        manager.detect_scenes(video)
        scenes = manager.get_scene_list()

        if not scenes:
            cap = cv2.VideoCapture(self.video_path)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fps = video.frame_rate
            scenes = [(FrameTimecode(0, fps=fps), FrameTimecode(duration, fps=fps))]

        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]

    def _compute_frame_at_time(self, cap, timestamp):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None, None, None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.downscale:
            img = cv2.resize(img, self.downscale, interpolation=cv2.INTER_AREA)
        text = pytesseract.image_to_string(img)
        hash = imagehash.phash(Image.fromarray(img))
        hash_vec = hash.hash.flatten().astype(np.uint8)
        return hash, hash_vec, text.strip()


    def _compute_frame(self, cap, start, end):
        mid_sec = (start + end) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            return (None, None, None)
        if frame is None:
            return (None, None, None)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.downscale:
            img = cv2.resize(img, self.downscale, interpolation=cv2.INTER_AREA)
        text = pytesseract.image_to_string(img)
        hash = imagehash.phash(Image.fromarray(img))
        hash_vec = hash.hash.flatten().astype(np.uint8)
        return (hash, hash_vec, text.strip())

    def _load_faiss_index(self, index_path, dim):
        return faiss.read_index(index_path) if os.path.exists(index_path) else faiss.IndexFlatIP(dim)

    def _save_faiss_index(self, index, indexPath):
        faiss.write_index(index, indexPath)
    
    @staticmethod
    def normalize_vec(vec):
        return normalize([vec])[0].astype(np.float32)

    def _text_embedding(self, texts):
        text = ' '.join([t for t in texts if t]).strip().lower()
        clean = re.sub(r"\s+", " ", text)
        return embedder.encode(clean).astype(np.float32)

    def _extract_audio(self):
        audio = AudioSegment.from_file(self.video_path).set_channels(1).set_frame_rate(SAMPLE_RATE)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
        if len(samples) == 0:
            raise ValueError("Audio stream is empty or missing.")
        samples /= np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else 1
        total_sec = round(len(samples) / SAMPLE_RATE, 1)
        win = int(SAMPLE_RATE * 0.5)
        active_sec = round(sum(np.std(samples[i:i + win]) > 0.01 for i in range(0, len(samples), win)) * 0.5, 1)
    
        mfcc_features = mfcc(samples, samplerate=SAMPLE_RATE, numcep=13)
        if mfcc_features.shape[0] == 0:
            raise ValueError("No MFCC features extracted.")
        mfcc_vec = np.mean(mfcc_features, axis=0).astype(np.float32)
        return mfcc_vec, total_sec, active_sec
    
    def process(self):
        if self.db.videos.find_one({"video_path": self.video_path}):
            print(f"⚠️ Skipping {self.video_path}, already in DB.")
            return
        start_time = time.time()
        os.makedirs(self.output_dir, exist_ok=True)
        audio_index = self._load_faiss_index(os.path.join(self.output_dir, "audio.index"), 13)
        video_index = self._load_faiss_index(os.path.join(self.output_dir, "video.index"), 64)
        text_index  = self._load_faiss_index(os.path.join(self.output_dir, "text.index"), 384)
        
        scenes = self._detect_scenes()
        print(f"scenes detected for {self.video_path}: {len(scenes)}")
        scene_hashes = []
        texts = []
        hash_vectors = []
        cap = cv2.VideoCapture(self.video_path)

        for start, end in scenes:
            # timestamps = [
            #     start + 0.5,
            #     (start + end) / 2,
            #     max(end - 0.5, start + 0.5)  # Ensure it's not before start
            # ]
            timestamps = [
                (start + end) / 2
            ]
            hashes = []
            for ts in timestamps:
                h, hvec, txt = self._compute_frame_at_time(cap, ts)
                if h is not None:
                    hashes.append(str(h))
                    hash_vectors.append(hvec)
                if txt:
                    texts.append(txt)

            scene_hashes.append({
                "start": start,
                "end": end,
                "hashes": hashes
            })

        cap.release()

        print(f"iCHECK all scenes computed")
        # Save to MongoDB
        doc = {
            "video_path": self.video_path,
            "duration": scene_hashes[-1]["end"] if scene_hashes else 0,
            "processed_at": datetime.utcnow(),
            "scene_hashes": scene_hashes
        }
        self.db.videos.insert_one(doc)
        print(f"✅ Inserted {self.video_path} with {len(scene_hashes)} scenes.")

        # === 2. Aggregate and Store Video pHash Vector ===
        if hash_vectors:
            video_vec = np.mean(hash_vectors, axis=0)
            norm_video_vec = self.normalize_vec(video_vec)
            video_index.add(norm_video_vec.reshape(1, -1))
        
        # === 3. Aggregate and Store Text Embedding Vector ===
        # === 3. Aggregate and Store Text Embedding Vector ===
        if texts:
            embeddings = self._text_embedding(texts)
        else:
            print(f"⚠️ No text found in {self.video_path}, using zero vector.")
            embeddings = np.zeros(384, dtype=np.float32)

        text_vec = self.normalize_vec(embeddings)
        text_index.add(np.array([text_vec], dtype=np.float32))

        
        silent_audio = False
        try:
            audio_vec, dur_total, dur_active = self._extract_audio()
        except Exception as ae:
            print(f"⚠️ Audio issue in {self.video_path}: {ae}")
            audio_vec = np.zeros(13, dtype=np.float32)
            dur_total, dur_active = 0.0, 0.0
            silent_audio = True

        norm_audio = self.normalize_vec(audio_vec)
        audio_index.add(np.array([norm_audio]))

        # === 5. Save All Indices and Metadata ===
        self._save_faiss_index(video_index, os.path.join(self.output_dir, "video.index"))
        self._save_faiss_index(text_index, os.path.join(self.output_dir, "text.index"))
        self._save_faiss_index(audio_index, os.path.join(self.output_dir, "audio.index"))
        print("Audio index count:", audio_index.ntotal)
        print("Video index count:", video_index.ntotal)
        print("Text index count: ", text_index.ntotal)

        meta_path = os.path.join(self.output_dir, META_PATH)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
        else:
            metadata = []

        metadata.append({
            "filename": os.path.basename(self.video_path),
            "video_path": self.video_path,
            "num_scenes": len(scene_hashes),
            "duration": doc["duration"],
            "silent_audio": silent_audio,
        })

        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        print("FAISS indices and metadata saved.")
        end_time = time.time()
        duration = end_time - start_time

        print(f"Time taken by {self.video_path}: {duration:.2f} seconds")

    def compute_vectors(self):
        scenes = self._detect_scenes()
        hash_vectors = []
        texts = []
        cap = cv2.VideoCapture(self.video_path)
        for start, end in scenes:
            result = self._compute_frame(cap, start, end)
            if result:
                _, hash_vec, text = result
                hash_vectors.append(hash_vec)
                if text:
                    texts.append(text)
        cap.release()
        video_vec = np.mean(hash_vectors, axis=0).astype(np.float32) if hash_vectors else np.zeros(64, dtype=np.float32)
        text_vec = self._text_embedding(texts) if texts else np.zeros(384, dtype=np.float32)

        try:
            audio_vec, _, _ = self._extract_audio()
            silent_audio = False
        except Exception:
            audio_vec = np.zeros(13, dtype=np.float32)
            silent_audio = True

        return {
            "video": self.normalize_vec(video_vec),
            "text": self.normalize_vec(text_vec),
            "audio": self.normalize_vec(audio_vec),
            "silent_audio": silent_audio
        }
        

