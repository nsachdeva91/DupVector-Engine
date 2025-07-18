from batch_processor import process_all_videos
from pymongo import MongoClient
from itertools import combinations
import os

# from scene_compare import SceneAddRemoveDetector  # Your class from previous step


def clear_existing_indices(index_dir="faiss_indices"):
    files_to_delete = ["audio.index", "video.index", "text.index", "metadata.pkl"]
    for filename in files_to_delete:
        path = os.path.join(index_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"🗑️ Deleted: {path}")

def removeDataFromDB():
    try:
        client = MongoClient()
        db = client["video_dedup"]  
        result = db.videos.delete_many({})

    except Exception as e:
        print(f"❌ Error connecting/deleting to MongoDB: {e}")
    
    clear_existing_indices()
# def batch_compare_all_pairs(db_name="video_dedup", use_faiss=False, tolerance=5, verbose=True):
#     db = MongoClient()[db_name]
#     videos = db.videos.distinct("video_path")
#     print(f"📂 Found {len(videos)} videos in DB.")

#     detector = SceneAddRemoveDetector(db=db, use_faiss=use_faiss, tolerance=tolerance)

#     for vid_a, vid_b in combinations(videos, 2):
#         if vid_a == vid_b:
#             continue

#         if verbose:
#             print(f"\n🔁 Comparing: {vid_a} ↔ {vid_b}")

#         removed, added = detector.compare_videos(vid_a, vid_b)

#         print(f"📊 Result {vid_a} → {vid_b}:")
#         print(f"  ❌ Removed: {len(removed)}")
#         print(f"  ➕ Added: {len(added)}")

# def compareHybrid(db_name="video_dedup"):
#     db = MongoClient()[db_name]
#     detector = SceneAddRemoveDetector(db, use_faiss=False)
#     results = detector.compare_hybrid("videos/watch5.mp4")
#     for r in results:
#         print(f"\nCompared with: {r['candidate']}")
#         print(f"Aggregated Distance: {r['distance']:.2f}")
#         print(f"❌ Removed Scenes: {len(r['removed_scenes'])}")
#         print(f"➕ Added Scenes: {len(r['added_scenes'])}")

if __name__ == "__main__":
    removeDataFromDB()
    process_all_videos("videos", downscale=(256, 256))
    # compareHybrid()
    # batch_compare_all_pairs(use_faiss=True, tolerance=5, verbose=True)
    
    # detector = SceneAddRemoveDetector(use_faiss=True, tolerance=5)
    # removed, added = detector.compare_videos("videos/videoA.mp4", "videos/videoAB.mp4")