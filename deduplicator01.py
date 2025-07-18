import streamlit as st
import os
import faiss
import pickle
import numpy as np
import librosa
import whisper
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from pair_similarity_analyzer import VideoSimilarityAnalyzer
from itertools import combinations
import time
from collections import defaultdict, Counter
import networkx as nx
from pyvis.network import Network
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎞️ Define your video directory (update path as needed)
VIDEO_DIR = r"C:\Users\neeraj_sachdeva\Desktop\LatestCode_Git\scene-matcher\videos"
INDEX_DIR = "faiss_indices"
META_PATH = "metadata.pkl"
model = whisper.load_model("base")

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv

    def groups(self):
        from collections import defaultdict
        result = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            result[root].append(i)
        return list(result.values())


class VideoDeduplicator:
    def __init__(self, index_dir):
        self.index_dir = index_dir
        self.audio = self._safe_load_faiss("audio.index", expected_dim=13)
        self.video = self._safe_load_faiss("video.index", expected_dim=64)
        self.text  = self._safe_load_faiss("text.index",  expected_dim=384)
        self.meta = self._load_meta()
        self.combined_index = None  # Will be created on demand

    def _safe_load_faiss(self, name, expected_dim):
        path = os.path.join(self.index_dir, name)
        index = faiss.read_index(path)
        vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])
        mask = np.linalg.norm(vectors, axis=1) > 1e-3
        # Zero out any all-zero vectors
        return np.where(mask[:, None], vectors, np.zeros((vectors.shape[0], expected_dim), dtype=np.float32))

    def _load_meta(self):
        with open(os.path.join(self.index_dir, META_PATH), "rb") as f:
            return pickle.load(f)

    def get_combined_vectors(self, weights=(1.0, 1.0, 1.0)):
        audio_w, video_w, text_w = weights
        audio = normalize(self.audio) * audio_w
        video = normalize(self.video) * video_w
        text  = normalize(self.text)  * text_w

        max_len = max(len(audio), len(video), len(text))

        def pad(arr, target_len, vec_dim):
            if len(arr) == target_len:
                return arr
            padding = np.zeros((target_len - len(arr), vec_dim), dtype=arr.dtype)
            return np.vstack([arr, padding])

        audio = pad(audio, max_len, audio.shape[1])
        video = pad(video, max_len, video.shape[1])
        text  = pad(text,  max_len, text.shape[1])

        return np.concatenate([audio, video, text], axis=1)


    def cluster_videos(self, eps=0.3, min_samples=2, weights=(1.0, 1.0, 1.0)):
        combined = self.get_combined_vectors(weights)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(combined)
        return clustering.labels_

    def _build_combined_faiss_index(self, vectors):
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    
    def check_duplicate(self, vec, weights=(1.0, 1.0, 1.0)):
        all_vectors = self.get_combined_vectors(weights=weights)
        vec = normalize([vec])[0]
        base_vectors = all_vectors[:-1]

        if self.combined_index is None or self.combined_index.ntotal != base_vectors.shape[0]:
            self.combined_index = self._build_combined_faiss_index(base_vectors)

        vec = np.ascontiguousarray(vec.astype(np.float32))
        faiss.normalize_L2(vec.reshape(1, -1))
        scores, indices = self.combined_index.search(vec.reshape(1, -1), k=1)
        return indices[0][0], scores[0][0]

    def find_duplicate_pairs(self, threshold=0.90, weights=(1.0, 1.0, 1.0), top_k=5):
        results = []

        # Unpack weights
        w_audio, w_video, w_text = weights

        # Normalize individual modality vectors
        norm_audio = normalize(self.audio)
        norm_video = normalize(self.video)
        norm_text  = normalize(self.text)

        for i in range(len(self.meta)):
            silent_i = self.meta[i].get("silent_audio", False)
            for j in range(i + 1, len(self.meta)):
                silent_j = self.meta[j].get("silent_audio", False)

                if silent_i or silent_j:
                    # Only use video + text
                    v_sim = np.dot(norm_video[i], norm_video[j])
                    t_sim = np.dot(norm_text[i], norm_text[j])
                    combined_sim = (w_video * v_sim + w_text * t_sim) / (w_video + w_text)
                else:
                    # Use audio + video + text
                    a_sim = np.dot(norm_audio[i], norm_audio[j])
                    v_sim = np.dot(norm_video[i], norm_video[j])
                    t_sim = np.dot(norm_text[i], norm_text[j])
                    combined_sim = (
                        w_audio * a_sim + w_video * v_sim + w_text * t_sim
                    ) / (w_audio + w_video + w_text)

                if combined_sim >= threshold - 1e-6:
                    results.append((i, j, combined_sim))

        return results

    def get_duplicate_groups(self, threshold=0.90, weights=(1.0, 1.0, 1.0), top_k=5):
        pairs = self.find_duplicate_pairs(threshold=threshold, weights=weights, top_k=top_k)
        uf = UnionFind(len(self.meta))
        for i, j, _ in pairs:
            uf.union(i, j)
        return uf.groups()

    def _cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) and np.linalg.norm(b) else 0.0


    def get_similarity_components(self, idx1, idx2=None, override_vector=None):
        if override_vector is not None:
            # Decompose combined override vector
            audio_len = self.audio.shape[1]
            video_len = self.video.shape[1]

            audio_vec = override_vector[:audio_len]
            video_vec = override_vector[audio_len:audio_len + video_len]
            text_vec  = override_vector[audio_len + video_len:]

            a1 = normalize([self.audio[idx1]])[0]
            v1 = normalize([self.video[idx1]])[0]
            t1 = normalize([self.text[idx1]])[0]

            a2 = normalize([audio_vec])[0]
            v2 = normalize([video_vec])[0]
            t2 = normalize([text_vec])[0]
        else:
            a1 = normalize([self.audio[idx1]])[0]
            v1 = normalize([self.video[idx1]])[0]
            t1 = normalize([self.text[idx1]])[0]

            a2 = normalize([self.audio[idx2]])[0]
            v2 = normalize([self.video[idx2]])[0]
            t2 = normalize([self.text[idx2]])[0]

        audio_sim = self._cosine_similarity(a1, a2)
        video_sim = self._cosine_similarity(v1, v2)
        text_sim  = self._cosine_similarity(t1, t2)

        return {
            "audio": round(float(audio_sim), 4),
            "video": round(float(video_sim), 4),
            "text":  round(float(text_sim), 4)
        }
        
def compute_audio_energy(file_path, silence_threshold=0.01, min_silence_duration=1.0):
    try:
        y, sr = librosa.load(file_path, sr=None)
        hop_length = 512
        frame_length = 1024
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        silent_segments = []
        silence_start = None
        for t, r in zip(times, rms):
            if r < silence_threshold:
                if silence_start is None:
                    silence_start = t
            elif silence_start is not None:
                duration = t - silence_start
                if duration >= min_silence_duration:
                    silent_segments.append((silence_start, t))
                silence_start = None
        return times, rms, silent_segments
    except Exception as e:
        st.warning(f"⚠️ Could not load audio from `{os.path.basename(file_path)}`: {e}")
        return np.array([]), np.array([]), []  # Return proper empty arrays

def plot_energy_comparison(file1, file2):
    times1, rms1, silence1 = compute_audio_energy(file1)
    times2, rms2, silence2 = compute_audio_energy(file2)

    if rms1.size == 0 or rms2.size == 0:
        st.info("📭 Could not extract audio energy from one or both files.")
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times1, rms1, label=os.path.basename(file1), alpha=0.7)
    ax.plot(times2, rms2, label=os.path.basename(file2), alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title("Audio Energy Comparison")
    ax.legend()

    for start, end in silence1:
        ax.axvspan(start, end, color='red', alpha=0.2, label='Silence in Video A')
    for start, end in silence2:
        ax.axvspan(start, end, color='blue', alpha=0.2, label='Silence in Video B')

    # Prevent duplicate legend entries for silent segments
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique_labels.append(l)
            unique_handles.append(h)
            seen.add(l)
    ax.legend(unique_handles, unique_labels)

    return fig

def get_transcription_and_language(file_path):
    try:
        result = model.transcribe(file_path, language=None)
        text = result["text"]
        lang = result["language"]
        return text, lang
    except Exception as e:
        st.warning(f"⚠️ Could not transcribe `{os.path.basename(file_path)}`: {e}")
        return "", ""
        
def get_text_similarity(text1, text2):
    vect = TfidfVectorizer().fit([text1, text2])
    tfidf = vect.transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score, 4)
        
# === STREAMLIT DASHBOARD ===
st.set_page_config(layout="wide")
st.title("📼 Video Deduplication Dashboard")
dedup = VideoDeduplicator(INDEX_DIR)

option = st.sidebar.selectbox("Choose Action", ["View Clusters",  "New upload"])

# if option == "View Clusters":
#     st.header("🔍 Grouped Duplicate Videos")
#     eps = st.slider("Clustering Sensitivity (lower = stricter)", 0.1, 0.5, 0.1, 0.05)
#     weights = 1.0, 1.0, 1.0  # Adjustable weight for testing
#     labels = dedup.cluster_videos(eps=eps, weights=weights)

#     clustered = {}
#     for label, meta in zip(labels, dedup.meta):
#         clustered.setdefault(label, []).append(meta["filename"])

#     for cluster_id, files in clustered.items():
#         if cluster_id == -1:
#             st.subheader("🧩 Distinct Videos")
#         else:
#             st.subheader(f"✅ Cluster {cluster_id + 1} (Variants)")
#         for f in files:
#             st.text(f)

# if option == "View Clusters":
#     st.header("🔍 Clustered Videos with Internal Similarity")
    
#     eps = st.slider("Clustering Sensitivity (lower = stricter)", 0.1, 0.5, 0.3, 0.05)
#     weights = 1.0, 1.0, 1.0  # Optional future slider
#     start_time = time.time()
#     labels = dedup.cluster_videos(eps=eps, weights=weights)

#     # Organize videos by cluster ID
#     clustered = {}
#     for label, meta in zip(labels, dedup.meta):
#         clustered.setdefault(label, []).append(meta["filename"])

#     analyzer = VideoSimilarityAnalyzer()

#     for cluster_id, files in clustered.items():
#         if cluster_id == -1:
#             st.subheader("🧩 Distinct Videos")
#             for f in files:
#                 st.text(f)
#         else:
#             st.subheader(f"✅ Cluster {cluster_id + 1} (Variants)")
#             # st.markdown("**Files:** " + ", ".join(files))
#             for f in files:
#                 st.text(f)
#             # Generate pairwise comparisons within the cluster
#             pairs = list(combinations(files, 2))
#             rows = []
#             for f1, f2 in pairs:
#                 i = analyzer.filenames.index(f1)
#                 j = analyzer.filenames.index(f2)
#                 a_sim = analyzer.cosine_similarity(analyzer.audio_vecs[i], analyzer.audio_vecs[j])
#                 v_sim = analyzer.cosine_similarity(analyzer.video_vecs[i], analyzer.video_vecs[j])
#                 t_sim = analyzer.cosine_similarity(analyzer.text_vecs[i], analyzer.text_vecs[j])
#                 label = analyzer.get_match_label(a_sim, v_sim, t_sim)
#                 rows.append([f1, f2, f"{a_sim:.4f}", f"{v_sim:.4f}", f"{t_sim:.4f}", label])

#             # Render table
#             if rows:
#                 df = pd.DataFrame(rows, columns=["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"])
#                 st.dataframe(df, use_container_width=True)
#     end_time = time.time()
#     duration = end_time - start_time
#     timeTakenText = f"Total Time taken for comparison: {duration:.2f} seconds"
#     st.text(timeTakenText)
category_colors = {
    "Duplicate": "#ffcccc",       # light red
    "Audio+Video": "#ccffcc",     # light green
    "Video+Text": "#cce5ff",      # light blue
    "Audio+Text": "#ffe5cc",      # light orange
}

def render_clustered_duplicate_groups_with_filters(dedup, eps=0.1, weights=(1.0, 1.0, 1.0)):
    start_time = time.time()

    st.header("🔍 Clustered Videos + Pairwise Similarity")

    labels = dedup.cluster_videos(eps=eps, weights=weights)
    clustered = {}
    for label, meta in zip(labels, dedup.meta):
        clustered.setdefault(label, []).append(meta["filename"])

    analyzer = VideoSimilarityAnalyzer()
    duplicate_graph = defaultdict(set)
    distinct_videos = set()
    results_per_cluster = {}

    allowed_categories = {"Duplicate", "Audio+Video", "Video+Text", "Audio+Text"}

    pairwise_labels_per_file = defaultdict(list)
    pairwise_scores_per_file = defaultdict(list)
    variant_label_priority = [
        "Audio+Video", "Video+Text", "Audio+Text",
        "Audio Only", "Video Only", "Text Only"
    ]
    variant_video_reason = {}

    for cluster_id, files in clustered.items():
        if cluster_id == -1:
            distinct_videos.update(files)
            continue

        pairs = list(combinations(files, 2))
        rows = []

        for f1, f2 in pairs:
            i = analyzer.filenames.index(f1)
            j = analyzer.filenames.index(f2)
            a_sim = analyzer.cosine_similarity(analyzer.audio_vecs[i], analyzer.audio_vecs[j])
            v_sim = analyzer.cosine_similarity(analyzer.video_vecs[i], analyzer.video_vecs[j])
            t_sim = analyzer.cosine_similarity(analyzer.text_vecs[i], analyzer.text_vecs[j])
            label = analyzer.get_match_label(a_sim, v_sim, t_sim)

            pairwise_labels_per_file[f1].append(label)
            pairwise_labels_per_file[f2].append(label)

            pairwise_scores_per_file[f1].append((f2, a_sim, v_sim, t_sim, label))
            pairwise_scores_per_file[f2].append((f1, a_sim, v_sim, t_sim, label))

            if label in allowed_categories:
                rows.append([
                    f1,
                    f2,
                    f"{a_sim * 100:.2f}%",
                    f"{v_sim * 100:.2f}%",
                    f"{t_sim * 100:.2f}%",
                    label
                ])

            if label == "Duplicate":
                duplicate_graph[f1].add(f2)
                duplicate_graph[f2].add(f1)

        results_per_cluster[cluster_id] = (rows, files)

    # Show clusters with filtered rows
    for cluster_id, (rows, files) in results_per_cluster.items():
        if not rows:
            continue

        df = pd.DataFrame(rows, columns=["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"])

        with st.expander(f"✅ Cluster {cluster_id + 1} ({len(df)} comparisons)", expanded=True):
            st.markdown("**Files in this cluster:**")
            for f in sorted(files):
                st.markdown(f"- {f}")

            def highlight_row_by_category(row):
                color = category_colors.get(row["Match Category"], "")
                return [f'background-color: {color}; color: black' if color else '' for _ in row]

            styled_df = df.style.apply(highlight_row_by_category, axis=1)
            st.dataframe(styled_df, use_container_width=True)

    # Use Union-Find to group duplicates
    filename_to_index = {f: i for i, f in enumerate(analyzer.filenames)}
    uf = UnionFind(len(analyzer.filenames))
    for f1, neighbors in duplicate_graph.items():
        for f2 in neighbors:
            uf.union(filename_to_index[f1], filename_to_index[f2])

    groups = []
    for group_indices in uf.groups():
        group = [analyzer.filenames[i] for i in group_indices]
        if len(group) > 1:
            groups.append(sorted(group))

    st.subheader("📂 Duplicate Groups")
    for i, group in enumerate(groups, 1):
        with st.expander(f"Group {i} ({len(group)} videos)", expanded=True):
            for f in sorted(group):
                st.markdown(f"- {f}")

    all_duplicates = set(v for g in groups for v in g)
    variant_videos = {}
    distinct_videos = set()

    for cluster_id, (rows, files) in results_per_cluster.items():
        if cluster_id == -1:
            continue

        for f in files:
            if f in all_duplicates:
                continue
            labels = pairwise_labels_per_file.get(f, [])
            if labels and all(l == "No Match" for l in labels):
                distinct_videos.add(f)
            else:
                # Assign best label and scores for tooltip
                for label in variant_label_priority:
                    if label in labels:
                        for (other_file, a_sim, v_sim, t_sim, l) in pairwise_scores_per_file[f]:
                            if l == label:
                                variant_videos[f] = {
                                    "reason": label,
                                    "score": f"Audio={a_sim*100:.2f}%, Video={v_sim*100:.2f}%, Text={t_sim*100:.2f}%",
                                    "color": category_colors.get(label, "")
                                }
                                break
                        break

    unclustered_videos = set()
    for label, files in clustered.items():
        if label == -1:
            unclustered_videos.update(files)

    # Show Variant Videos
    st.subheader("📁 Variant Videos")
    if variant_videos:
        with st.expander(f"{len(variant_videos)} Variant Files", expanded=True):
            for f, info in sorted(variant_videos.items()):
                reason = info["reason"]
                score = info["score"]
                color = info["color"]
                st.markdown(
                    f"<div style='background-color:{color}; color:black; padding:5px; border-radius:4px;'>"
                    f"🔹 <b>{f}</b><br/><small>Matched on: <i>{reason}</i> — {score}</small></div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No variant videos found.")

    # Show Distinct Videos (clustered but no matches)
    st.subheader("📂 Distinct Videos (in clusters with no matches)")
    if distinct_videos:
        with st.expander(f"{len(distinct_videos)} Distinct Files", expanded=True):
            for f in sorted(distinct_videos):
                st.markdown(
                    f"<div style='background-color:#ddd; color:black; padding:5px; border-radius:4px;'>"
                    f"🟦 <b>{f}</b> — No matching features in cluster</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No distinct videos found in clusters.")

    # Show Unclustered Distinct Videos
    st.subheader("📂 Unclustered Distinct Videos")
    if unclustered_videos:
        with st.expander(f"{len(unclustered_videos)} Unclustered Files", expanded=True):
            for f in sorted(unclustered_videos):
                st.markdown(
                    f"<div style='background-color:#f5f5f5; color:black; padding:5px; border-radius:4px;'>"
                    f"📄 <b>{f}</b> — Not clustered with any group</div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No unclustered distinct videos found.")

    st.markdown(f"⏱ **Total processing time:** `{time.time() - start_time:.2f} seconds`")



if option == "View Clusters":
    eps = st.slider("Clustering Sensitivity (lower = stricter)", 0.05, 0.5, 0.25, 0.05)
    render_clustered_duplicate_groups_with_filters(dedup, eps=eps)

elif option == "New upload":
    st.header("🧪 Check If a Video is a Duplicate")

    video_path = st.text_input("Enter full path to video file:", value="Sample_Good_AV_720p.mp4")
    start_time = time.time()

    if video_path and os.path.isfile(video_path):
        st.text("Processing selected video...")
        from video_processor import VideoProcessor
        from pymongo import MongoClient

        db = MongoClient()["video_dedup"]
        vp = VideoProcessor(db, video_path, output_dir=INDEX_DIR)
        vecs = vp.compute_vectors()

        is_silent = vecs["silent_audio"]
        audio_weight = 0.0 if is_silent else 1.0

        if is_silent:
            st.warning("⚠️ This video has no audio — only comparing text and video.")

        combined = np.concatenate([
            vecs["audio"] * audio_weight,
            vecs["video"],
            vecs["text"]
        ]).astype(np.float32)

        combined_norm = combined / np.linalg.norm(combined)
        combined_vectors = dedup.get_combined_vectors(weights=(audio_weight, 1.0, 1.0))
        combined_vectors_norm = combined_vectors / np.linalg.norm(combined_vectors, axis=1, keepdims=True)

        sims = combined_vectors_norm @ combined_norm
        k = st.slider("Number of matches", 1, 10, 5, 1)
        top_k_idx = np.argsort(-sims)[:k]
        top_k_sims = sims[top_k_idx]

        analyzer = VideoSimilarityAnalyzer()
        st.subheader(f"🔗 Top {k} Matching Videos")

        results = []
        distinct = []
        variants = []
        duplicates = []

        for rank, (idx, sim) in enumerate(zip(top_k_idx, top_k_sims), start=1):
            meta = dedup.meta[idx]
            filename = meta["filename"]
            similarities = dedup.get_similarity_components(idx, idx2=None, override_vector=combined)

            a_sim = similarities['audio']
            v_sim = similarities['video']
            t_sim = similarities['text']
            label = analyzer.get_match_label(a_sim, v_sim, t_sim)
            normalized_label = label.strip().lower()

            full_match_path = os.path.join(VIDEO_DIR, filename)

            show_audio_plot = normalized_label in ["video", "video+text"]
            show_language_analysis = normalized_label in ["video", "video+text"]
            show_transcript_analysis = normalized_label in ["audio+video", "video"]

            if show_audio_plot and os.path.isfile(full_match_path) and os.path.isfile(video_path):
                st.markdown(f"🎧 **Audio Energy Comparison with `{filename}`**")
                fig = plot_energy_comparison(video_path, full_match_path)
                if fig:
                    st.pyplot(fig)

            if show_language_analysis and os.path.isfile(full_match_path) and os.path.isfile(video_path):
                st.markdown(f"🈶 **Language Analysis with `{filename}`**")
                text_a, lang_a = get_transcription_and_language(video_path)
                text_b, lang_b = get_transcription_and_language(full_match_path)
                st.write(f"Uploaded video language: `{lang_a}`")
                st.write(f"Matched video language: `{lang_b}`")
                if lang_a != lang_b:
                    st.warning("⚠️ Audio language mismatch detected")

            if show_transcript_analysis and os.path.isfile(full_match_path) and os.path.isfile(video_path):
                st.markdown(f"📝 **Transcript Comparison with `{filename}`**")
                text_a, _ = get_transcription_and_language(video_path)
                text_b, _ = get_transcription_and_language(full_match_path)

                if text_a and text_b:
                    similarity_score = get_text_similarity(text_a, text_b)
                    st.write(f"Transcript similarity with `{filename}`: `{similarity_score * 100:.2f}%`")
                    if similarity_score < 0.5:
                        st.warning("⚠️ Spoken text content shows low similarity — possible dialogue change or dubbing")
                else:
                    st.info("📭 Could not extract transcripts from uploaded video or matched video.")

            elif normalized_label in ["video", "video+text"] and not os.path.isfile(full_match_path):
                st.warning("❗ One or both video files are missing from VIDEO_DIR")

            result_row = [
                rank,
                filename,
                f"{a_sim * 100:.2f}%",
                f"{v_sim * 100:.2f}%",
                f"{t_sim * 100:.2f}%",
                f"{sim * 100:.2f}%",
                label
            ]
            results.append(result_row)

            if sim >= 0.9:
                duplicates.append(result_row)
            elif sim >= 0.7:
                variants.append(result_row)
            else:
                distinct.append(result_row)

        
                
        # Show main results table
        df = pd.DataFrame(results, columns=[
            "Rank", "Filename", "AudioSim", "VideoSim", "TextSim", "OverallSim", "Match Category"
        ]) 

        # Optional: Color styling for the match categories
        def highlight_row_by_category(row):
            color = category_colors.get(row["Match Category"], "")
            return [f'background-color: {color}; color: black' if color else '' for _ in row]

        st.dataframe(df.style.apply(highlight_row_by_category, axis=1), use_container_width=True)

        # Display distinct, variants, and duplicates with similarity scores
        st.subheader("📊 Video Categorization")
        st.write("### Distinct Videos (Low similarity)")
        if distinct:
            st.write(f"Found {len(distinct)} distinct videos:")
            st.dataframe(pd.DataFrame(distinct, columns=[
                "Rank", "Filename", "AudioSim", "VideoSim", "TextSim", "OverallSim", "Match Category"
            ]))
        
        st.write("### Variant Videos (Moderate similarity)")
        if variants:
            st.write(f"Found {len(variants)} variant videos:")
            st.dataframe(pd.DataFrame(variants, columns=[
                "Rank", "Filename", "AudioSim", "VideoSim", "TextSim", "OverallSim", "Match Category"
            ]))
        
        st.write("### Duplicate Videos (High similarity)")
        if duplicates:
            st.write(f"Found {len(duplicates)} duplicate videos:")
            st.dataframe(pd.DataFrame(duplicates, columns=[
                "Rank", "Filename", "AudioSim", "VideoSim", "TextSim", "OverallSim", "Match Category"
            ]))

    else:
        st.warning("Please enter a valid path")
    
    st.markdown(f"⏱ **Total processing time:** `{time.time() - start_time:.2f} seconds`")


# if option == "View Clusters":
#     st.header("🔍 Clustered Duplicate Videos with Pairwise Similarity")

#     eps = st.slider("Clustering Sensitivity (lower = stricter)", 0.1, 0.5, 0.3, 0.05)
#     start_time = time.time()
#     weights = 1.0, 1.0, 1.0
#     labels = dedup.cluster_videos(eps=eps, weights=weights)

#     clustered = {}
#     for label, meta in zip(labels, dedup.meta):
#         clustered.setdefault(label, []).append(meta["filename"])

#     analyzer = VideoSimilarityAnalyzer()

#     duplicate_graph = defaultdict(set)
#     distinct_videos = set()

#     for cluster_id, files in clustered.items():
#         if cluster_id == -1:
#             st.subheader("🧩 Distinct Videos (Outliers)")
#             for f in files:
#                 st.text(f)
#             distinct_videos.update(files)
#             continue

#         st.subheader(f"✅ Cluster {cluster_id + 1}")
#         for f in files:
#             st.text(f)
#         # Intra-cluster similarity
#         pairs = list(combinations(files, 2))
#         rows = []
#         for f1, f2 in pairs:
#             i = analyzer.filenames.index(f1)
#             j = analyzer.filenames.index(f2)
#             a_sim = analyzer.cosine_similarity(analyzer.audio_vecs[i], analyzer.audio_vecs[j])
#             v_sim = analyzer.cosine_similarity(analyzer.video_vecs[i], analyzer.video_vecs[j])
#             t_sim = analyzer.cosine_similarity(analyzer.text_vecs[i], analyzer.text_vecs[j])
#             label = analyzer.get_match_label(a_sim, v_sim, t_sim)
#             rows.append([f1, f2, f"{a_sim:.4f}", f"{v_sim:.4f}", f"{t_sim:.4f}", label])

#             if label == "Duplicate":
#                 duplicate_graph[f1].add(f2)
#                 duplicate_graph[f2].add(f1)

#         if rows:
#             df = pd.DataFrame(rows, columns=["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"])
#             st.dataframe(df, use_container_width=True)

#     # === Generate Duplicate Groups from duplicate_graph
#     visited = set()
#     groups = []

#     def dfs(node, group):
#         if node in visited: return
#         visited.add(node)
#         group.append(node)
#         for neighbor in duplicate_graph[node]:
#             dfs(neighbor, group)

#     for node in duplicate_graph:
#         if node not in visited:
#             group = []
#             dfs(node, group)
#             groups.append(sorted(group))

#     # === Show Groups
#     st.subheader("📂 Duplicate Groups (Cluster + Similarity Based)")
#     if groups:
#         for i, group in enumerate(groups, 1):
#             st.markdown(f"**Group {i}:** {', '.join(group)}")
#     else:
#         st.info("No duplicate groups found in clustered videos.")

#     # === Show Remaining Distinct Videos (not in any duplicate group)
#     all_duplicates = set(v for group in groups for v in group)
#     for label, files in clustered.items():
#         if label != -1:
#             for f in files:
#                 if f not in all_duplicates:
#                     distinct_videos.add(f)

#     st.subheader("📁 Distinct Videos (Clustered but Not Matched)")
#     if distinct_videos:
#         for f in sorted(distinct_videos):
#             st.markdown(f"- {f}")
#     else:
#         st.info("All clustered videos belong to duplicate groups.")
    
#     end_time = time.time()
#     duration = end_time - start_time
#     timeTakenText = f"Total Time taken for comparison: {duration:.2f} seconds"
#     st.subheader(timeTakenText)





elif option == "Batch Deduplication Report":
    st.header("Similar Video Pairs")
    threshold = st.slider("Similarity threshold", 0.95, 1.0, 0.98, 0.01)
    max_pairs = st.number_input("Max pairs to display", 1, 1000, 100)
    
    with st.spinner("Scanning for similar pairs..."):
        pairs = dedup.find_duplicate_pairs(threshold=threshold)
    
    st.success(f"Found {len(pairs)} similar pairs")

    for i, (idx1, idx2, sim) in enumerate(sorted(pairs, key=lambda x: -x[2])[:max_pairs]):
        meta1 = dedup.meta[idx1]
        meta2 = dedup.meta[idx2]
        similarities = dedup.get_similarity_components(idx1, idx2)

        st.markdown(f"""
        🔁 **{meta1['filename']}** ↔ **{meta2['filename']}**  
        **Combined Similarity:** `{sim:.4f}`  
        - 🎧 Audio: `{similarities['audio']:.4f}`  
        - 🖼️ Video: `{similarities['video']:.4f}`  
        - 📝 Text: `{similarities['text']:.4f}`
        """)
        if dedup.meta[idx1]["silent_audio"] or dedup.meta[idx2]["silent_audio"]:
            st.warning("⚠️ One or both videos have no audio — audio similarity may be meaningless.")

    st.header("Duplicate Groups & Distinct Videos")
    groups = dedup.get_duplicate_groups(threshold=threshold)

    duplicates = [g for g in groups if len(g) > 1]
    singles = [g[0] for g in groups if len(g) == 1]

    st.subheader(f"✅ {len(duplicates)} Duplicate Groups")
    for i, group in enumerate(duplicates):
        st.markdown(f"**Group {i + 1}**:")
        for idx in group:
            st.text(f"📼 {dedup.meta[idx]['filename']}")
        st.markdown("---")

    st.subheader(f"🧩 {len(singles)} Distinct Videos")
    for idx in singles:
        st.text(f"📼 {dedup.meta[idx]['filename']}")


elif option == "Pair similarity detailed":
    
    st.title("🎥 Video Similarity & Deduplication Dashboard")
    st.markdown("This compares all videos and groups duplicates using audio, video, and text similarity.")

    analyzer = VideoSimilarityAnalyzer()

    progress_bar = st.progress(0.0, text="Starting...")
    analyzer.run_comparison(progress_callback=lambda p, msg: progress_bar.progress(p, text=msg))
    progress_bar.empty()
    st.success("✅ Comparison complete.")

    # Table
    st.subheader("📋 Pairwise Comparison Results")
    df = pd.DataFrame(analyzer.get_results_table(), columns=["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"])
    st.dataframe(df, use_container_width=True)

    # Download CSV
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="video_similarity_results.csv")

    # Category Distribution
    st.subheader("📊 Match Category Distribution")
    counts = analyzer.get_category_counts()
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors = ["#4CAF50" if "Match" in l else "#F44336" for l in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom')
    st.pyplot(fig)

    # Duplicate Groups
    st.subheader("📂 Duplicate Groups")
    groups = analyzer.get_duplicate_groups()
    if groups:
        for i, group in enumerate(groups, 1):
            st.markdown(f"**Group {i}:** {', '.join(group)}")
    else:
        st.info("No duplicate groups found.")

    # Distinct Videos
    st.subheader("📁 Distinct Videos")
    distinct = analyzer.get_distinct_files()
    if distinct:
        for f in distinct:
            st.markdown(f"- {f}")
    else:
        st.info("All videos are part of duplicate groups.")

elif option == "Pair similarity detailed":
    
    st.title("🎥 Video Similarity & Deduplication Dashboard")
    st.markdown("This compares all videos and groups duplicates using audio, video, and text similarity.")

    analyzer = VideoSimilarityAnalyzer()

    progress_bar = st.progress(0.0, text="Starting...")
    analyzer.run_comparison(progress_callback=lambda p, msg: progress_bar.progress(p, text=msg))
    progress_bar.empty()
    st.success("✅ Comparison complete.")

    # Table
    st.subheader("📋 Pairwise Comparison Results")
    df = pd.DataFrame(analyzer.get_results_table(), columns=["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"])
    st.dataframe(df, use_container_width=True)

    # Download CSV
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="video_similarity_results.csv")

    # Category Distribution
    st.subheader("📊 Match Category Distribution")
    counts = analyzer.get_category_counts()
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors = ["#4CAF50" if "Match" in l else "#F44336" for l in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom')
    st.pyplot(fig)

    # Duplicate Groups
    st.subheader("📂 Duplicate Groups")
    groups = analyzer.get_duplicate_groups()
    if groups:
        for i, group in enumerate(groups, 1):
            st.markdown(f"**Group {i}:** {', '.join(group)}")
    else:
        st.info("No duplicate groups found.")

    # Distinct Videos
    st.subheader("📁 Distinct Videos")
    distinct = analyzer.get_distinct_files()
    if distinct:
        for f in distinct:
            st.markdown(f"- {f}")
    else:
        st.info("All videos are part of duplicate groups.")
#    else:
#        st.error("❌ One or both file paths are invalid.")
