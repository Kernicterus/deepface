import cv2
import mediapipe as mp
import numpy as np
import os
from itertools import pairwise

from deepface import DeepFace
from utils import get_video_rotation, rotate_frame, generate_random_string
from face_utils import load_face_data, save_face_img, upd_csv_video_faces, align_face, check_face_quality
from mtcnn import MTCNN

RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"

MODEL="Facenet512"
SIMILARITY_THRESHOLD = 0.45
DETECTOR_BACKEND = "mtcnn"
OUTPUT_FACES_FOLDER = "./faces_detected"

class FaceData:
    def __init__(self, face, embedding, confidence, face_w, face_h):
        self.face = face
        self.embedding = embedding
        self.confidence = confidence
        self.size = face_w * face_h
        self.is_new = True
        self.match_old_index = None
        self.is_better_match = False
        self.filename = None

def remove_duplicate_faces(faces_detected: list[FaceData]) -> list[FaceData]:
    to_remove = set()

    for i in range(len(faces_detected) - 1):
        if i in to_remove:
            continue

        for j in range(i + 1, len(faces_detected)):
            if j in to_remove:
                continue

            result = DeepFace.verify(
                faces_detected[i].embedding,
                faces_detected[j].embedding,
                model_name=MODEL,
                distance_metric="cosine",
                silent=True,
                threshold=SIMILARITY_THRESHOLD,
                detector_backend="skip",
                enforce_detection=False
            )

            if result["verified"]:
                if faces_detected[i].confidence > faces_detected[j].confidence:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break

        faces_detected = [face for index, face in enumerate(faces_detected) if index not in to_remove]

    return faces_detected


# def upd_face_data(is_new, match_old_index, best_match_old_index, face, old_faces_data, faces_in_video_processed, face_params):
#     """
#     Save face data and image based on whether it's new or updating existing.
    
#     Args:
#         is_new (bool): Whether this is a new face
#         best_match_old_index (int): Index of best matching face if not new
#         face: Face image array
#         old_faces_data (list): Existing face data
#         faces_in_video_processed (dict): Dictionary of faces found in current video and their embeddings
#         face_params (dict): Dictionary of face parameters including embedding, confidence, face_w, face_h
#     """
#     global IMAGES_IN_FOLDER
#     if is_new:
#         id_new_img = IMAGES_IN_FOLDER
#         IMAGES_IN_FOLDER += 1
#         face_filename = f"face_{id_new_img}.jpg"
#         faces_in_video_processed[face_filename] = [face, True]
#         old_faces_data.append((face_filename, face_params[0], face_params[1], face_params[2]))

#     elif best_match_old_index is not None:
#         old_face_filename = old_faces_data[best_match_old_index][0]
#         faces_in_video_processed[old_face_filename] = [face, True]
#         old_faces_data[best_match_old_index] = (old_face_filename, face_params[0], face_params[1], face_params[2])

#     else:
#         if match_old_index is not None:
#             old_face_filename = old_faces_data[match_old_index][0]
#             faces_in_video_processed.setdefault(old_face_filename, [face, False])


# def find_matching_face(face_params, old_faces_data):
#     """
#     Find if the face matches any existing faces in database.
    
#     Args:
#         face_params (list): Face parameters including embedding, confidence, face_w * face_h
#         old_faces_data (list): Existing face data
#     Returns:
#         tuple: (is_new, best_match_old_index, match_old_index) 
#     """
#     is_new = True
#     best_match_old_index = None
#     match_old_index = None
#     for i, (filename, known_embedding, old_confidence, old_size) in enumerate(old_faces_data):
#         try:
#             result = DeepFace.verify(face_params[0], known_embedding, model_name=MODEL, distance_metric="cosine", silent=True, threshold=SIMILARITY_THRESHOLD, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
#             # if result["verified"]:
#             #     print(f"{GREEN}result: {result['verified']} - time processing: {result['time']} - distance: {result['distance']}{RESET}")
#             # else:
#             #     print(f"{RED}result: {result['verified']} - time processing: {result['time']} - distance: {result['distance']}{RESET}")
#             if result["verified"]:
#                 is_new = False
#                 match_old_index = i
#                 if face_params[1] > old_confidence:
#                     best_match_old_index = i
#                 break
#         except Exception as e:
#             print(f"Error: {e}")
#             return False, None, None
                
#     return is_new, best_match_old_index, match_old_index

# def process_detection(detection, frame, h, w, faces_in_video_processed, old_faces_data):
#     """
#     Process a single face detection.
    
#     Args:
#         detection: MediaPipe face detection
#         frame: Video frame
#         h, w: Frame dimensions
#         detection_count (int): Running detection count
#         face_data (list): Existing face data
#         faces_in_video_processed (list): List of faces found in current video
#         output_folder (str): Folder to save face images
#     """

#     bboxC = detection.location_data.relative_bounding_box
#     x = max(0, int(bboxC.xmin * w))
#     y = max(0, int(bboxC.ymin * h))
#     face_w = int(bboxC.width * w)
#     face_h = int(bboxC.height * h)

#     face = frame[y:y + face_h, x:x + face_w]
#     confidence = detection.score[0]
#     detector = MTCNN()

#     if not check_face_quality(face, detector):
#         return

#     if face.shape[0] > 0 and face.shape[1] > 0:
#         try:
#             # face_aligned = align_face(face, detector)
#             embedding = DeepFace.represent(face, model_name=MODEL, enforce_detection=False, detector_backend=DETECTOR_BACKEND)[0]["embedding"]
#             face_params = [embedding, confidence, face_w * face_h]
#             is_new, best_match_old_index, match_old_index = find_matching_face(face_params, old_faces_data)
#             upd_face_data(is_new, match_old_index, best_match_old_index, face, old_faces_data, faces_in_video_processed, face_params)

#         except Exception as e:
#             print(f"Error : {e}")

# def process_frame(frame, face_detection, h, w, faces_in_video_processed, old_faces_data):
#     """
#     Process a single video frame to detect and analyze faces.
    
#     Args:
#         frame: Video frame to process
#         face_detection: MediaPipe face detection model
#         h, w: Frame dimensions
#         faces_in_video_processed (list): List of faces found in current video
#         output_folder (str): Folder to save face images
#         old_faces_data (list): Existing face data
#     """
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(frame_rgb)

#     if results.detections:
#         for detection in results.detections:
#             process_detection(detection, frame, h, w, faces_in_video_processed, old_faces_data)


def extract_face_datas(detection, frame, detector) -> FaceData:
    """
    Extract face data from a detected face.
    
    Args:
        detection: MediaPipe face detection processed
        frame: Video frame
        detector: face detector
    Returns:
        FaceData: Face data
    """
    try:
        h, w, _ = frame.shape
        bboxC = detection.location_data.relative_bounding_box
        x = max(0, int(bboxC.xmin * w))
        y = max(0, int(bboxC.ymin * h))
        face_w = int(bboxC.width * w)
        face_h = int(bboxC.height * h)

        face = frame[y:y + face_h, x:x + face_w]
        confidence = detection.score[0]

        if  face.shape[0] > 0 and face.shape[1] > 0 and check_face_quality(face, detector):
            # face_aligned = align_face(face, detector)
            embedding = DeepFace.represent(face, model_name=MODEL, enforce_detection=False, detector_backend="skip")[0]["embedding"]
            return FaceData(face, embedding, confidence, face_w, face_h)

    except Exception as e:
        print(f"{RED}Error : {e}{RESET}")
        return None
        

def detect_faces_in_video(video, rotation, frame_skip) -> list:
    """
    Process video frames to detect faces.
    
    Args:
        video: OpenCV video capture object
        rotation (int): Video rotation angle
        frame_skip (int): Number of frames to skip between processing
        
    Returns:
        list: List of face data found in this video
    """

    frame_count = 0
    mp_face_detection = mp.solutions.face_detection
    faces_detected = []
    detector = MTCNN()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            frame = rotate_frame(frame, rotation)

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    face_data = extract_face_datas(detection, frame, detector)
                    if face_data is not None:
                        faces_detected.append(face_data)

    return faces_detected

def verify_faces_detected(faces_detected: list[FaceData], old_faces_data: list) -> list[FaceData]:
    """
    Process detected faces and update face data.
    
    Args:
        faces_detected (list): List of FaceData objects
        old_faces_data (list): List of old face data
    Returns:
        dict: Dictionary of faces found in this video
    """
    if len(faces_detected) == 0:
        return []
    
    faces_detected = remove_duplicate_faces(faces_detected)

    for face_data in faces_detected:
        for i, (filename, known_embedding, old_confidence, old_size) in enumerate(old_faces_data):
            try:
                result = DeepFace.verify(
                    face_data.embedding,
                    known_embedding,
                    model_name=MODEL,
                    distance_metric="cosine",
                    silent=True,
                    threshold=SIMILARITY_THRESHOLD,
                    detector_backend="skip",
                    enforce_detection=False
                    )
                if result["verified"]:
                    face_data.is_new = False
                    face_data.match_old_index = i
                    face_data.filename = filename
                    if face_data.confidence > old_confidence:
                        face_data.is_better_match = True
                    break
            except Exception as e:
                print(f"Error: {e}")
    
    return faces_detected

def process_faces_detected(faces_detected: list[FaceData], old_faces_data: list) -> dict:
    """
    Process detected faces and update face data.
    
    Args:
        faces_detected (list): List of FaceData objects
        old_faces_data (list): List of old face data
    Returns:
        dict: Dictionary of faces found in this video
    """
    faces_in_processed_video = {}

    if len(faces_detected) == 0:
        return {}
    
    for face_data in faces_detected:
        if face_data.is_new and not face_data.filename:
            face_data.filename = f"{generate_random_string(16)}.jpg"
        faces_in_processed_video[face_data.filename] = [face_data.face, face_data.is_new or face_data.is_better_match]
        old_faces_data.append((face_data.filename, face_data.embedding, face_data.confidence, face_data.size))

    return faces_in_processed_video


def extract_faces(video_path, frame_skip=10, old_faces_data_path=None, csv_face_by_video_path=None):
    """
    Extract and process faces from a video file, saving unique faces and their embeddings.
    
    Args:
        video_path (str): Path to the video file
        frame_skip (int): Number of frames to skip between processing
        old_faces_data_path (str): Path where face embeddings data are saved
        csv_face_by_video_path (str): Path to save face-video mapping CSV
    """

    if old_faces_data_path is None:
        old_faces_data_path = "face_data.npy"
    if csv_face_by_video_path is None:
        csv_face_by_video_path = "face_by_video.csv"

    video_name = os.path.basename(video_path)
    old_faces_data = load_face_data(old_faces_data_path)
    rotation = get_video_rotation(video_path)

    video = cv2.VideoCapture(video_path)
    output_folder = OUTPUT_FACES_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    faces_detected = detect_faces_in_video(video, rotation, frame_skip)
    faces_detected_unique = verify_faces_detected(faces_detected, old_faces_data)
    faces_in_processed_video = process_faces_detected(faces_detected_unique, old_faces_data)

    video.release()
    cv2.destroyAllWindows()

    upd_csv_video_faces(faces_in_processed_video, video_name, csv_face_by_video_path)
    save_face_img(faces_in_processed_video, output_folder)
    np.save(old_faces_data_path, np.array(old_faces_data, dtype=object), allow_pickle=True)


def main():
    try:
        video_path = "./videos/VID2.mp4"
        extract_faces(video_path, 10)
    except Exception as e:
        print(f"{RED}Error : {e}{RESET}")


if __name__ == "__main__":
    main()
