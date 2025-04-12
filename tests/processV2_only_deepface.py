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
SIMILARITY_THRESHOLD = 0.5
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
    """
    Remove duplicate faces from a list of detected faces by comparing embeddings.
    
    Args:
        faces_detected (list): List of FaceData objects
    Returns:
        list: List of unique face data
    """
    if len(faces_detected) <= 1:
        return faces_detected

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

    print(f"{RED}Removing {len(to_remove)} faces{RESET}")
    faces_detected = [face for index, face in enumerate(faces_detected) if index not in to_remove]
    print(f"{YELLOW}Faces detected after removing duplicates : {len(faces_detected)}{RESET}")

    return faces_detected


def extract_face_datas(detection, frame) -> FaceData:
    """
    Extract face data from a detected face and record them in a FaceData object.

    Args:
        detection: MediaPipe face detection processed
        frame: Video frame
        detector: face detector
    Returns:
        FaceData: Face data object
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

        if  face.shape[0] > 0 and face.shape[1] > 0 and check_face_quality(face):
            # face_aligned = align_face(face, detector)
            embedding = DeepFace.represent(face, model_name=MODEL, enforce_detection=False, detector_backend="skip")[0]["embedding"]
            return FaceData(face, embedding, confidence, face_w, face_h)

    except Exception as e:
        print(f"{RED}Error : {e}{RESET}")
        return None
        
def detect_faces_in_video(video : cv2.VideoCapture, rotation : int, frame_skip : int) -> list[FaceData]:
    """
    Process video frames to detect faces.
    
    Args:
        video: OpenCV video capture object
        rotation (int): Video rotation angle
        frame_skip (int): Number of frames to skip between processing
        
    Returns:
        list: List of FaceData objects found in this video
    """

    frame_count = 0
    faces_detected = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        frame = rotate_frame(frame, rotation)

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        try:
            analysis = DeepFace.represent(frame, model_name=MODEL, enforce_detection=False, detector_backend=DETECTOR_BACKEND)
            for detection in analysis:
                embedding = np.array(detection["embedding"])
                region = detection["facial_area"]
                confidence = detection["face_confidence"]
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                face = frame[y:y + h, x:x + w]
                if  face.shape[0] > 0 and face.shape[1] > 0 and check_face_quality(face):
                    faces_detected.append(FaceData(face, embedding, confidence, w, h))

        except Exception as e:
            print(f"{RED}DeepFace.represent error : {e}{RESET}")
            continue

    return faces_detected

# def detect_faces_in_video(video : cv2.VideoCapture, rotation : int, frame_skip : int) -> list[FaceData]:
#     """
#     Process video frames to detect faces.
    
#     Args:
#         video: OpenCV video capture object
#         rotation (int): Video rotation angle
#         frame_skip (int): Number of frames to skip between processing
        
#     Returns:
#         list: List of FaceData objects found in this video
#     """

#     frame_count = 0
#     mp_face_detection = mp.solutions.face_detection
#     faces_detected = []
#     detector = MTCNN()

#     with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break
            
#             frame = rotate_frame(frame, rotation)

#             frame_count += 1
#             if frame_count % frame_skip != 0:
#                 continue

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_detection.process(frame_rgb)

#             if results.detections:
#                 for detection in results.detections:
#                     face_data = extract_face_datas(detection, frame, detector)
#                     if face_data is not None:
#                         faces_detected.append(face_data)
#     return faces_detected

def verify_faces_detected(faces_detected: list[FaceData], old_faces_data: list) -> list[FaceData]:
    """
    Verify faces detected by comparing embeddings to old faces data
    If the face is not a new face, confidences are compared to determine if the face is a more confident match than the old one.

    Args:
        faces_detected (list): List of FaceData objects
        old_faces_data (list): List of old face data
    Returns:
        list: List of FaceData objects updated
    """
    if len(faces_detected) == 0:
        return []
    
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


def process_verified_faces(faces_detected: list[FaceData], old_faces_data: list) -> dict:
    """
    Process verified faces and update face data.
    If the face is a new face, it is added to the embeddings list.
    If the face is not a new face but has a better confidence, the old face is updated in the embeddings list.
    
    Args:
        faces_detected (list): List of FaceData objects
        old_faces_data (list): List of old face data
    Returns:
        dict: Dictionary of faces found in this video with a boolean indicating if the face must be saved/updated
    """
    faces_in_processed_video = {}

    if len(faces_detected) == 0:
        return {}
    
    for face_data in faces_detected:
        if face_data.is_new and not face_data.filename:
            face_data.filename = f"{generate_random_string(16)}.jpg"
            old_faces_data.append((face_data.filename, face_data.embedding, face_data.confidence, face_data.size))
        elif face_data.is_better_match:
            old_faces_data[face_data.match_old_index] = (face_data.filename, face_data.embedding, face_data.confidence, face_data.size)
        faces_in_processed_video[face_data.filename] = [face_data.face, face_data.is_new or face_data.is_better_match]

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
    print(f"{GREEN}Faces detected : {len(faces_detected)}{RESET}")

    faces_detected_unique = remove_duplicate_faces(faces_detected)
    print(f"{GREEN}Faces detected without duplicates : {len(faces_detected_unique)}{RESET}")

    faces_detected_datas_updated = verify_faces_detected(faces_detected_unique, old_faces_data)
    print(f"{GREEN}New faces detected : {len([face for face in faces_detected_datas_updated if face.is_new])}{RESET}")

    faces_in_processed_video = process_verified_faces(faces_detected_datas_updated, old_faces_data)
    print(f"{GREEN}Faces in processed video : {len(faces_in_processed_video)}{RESET}")

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
