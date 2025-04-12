import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

from deepface import DeepFace
from utils import get_video_rotation, rotate_frame, generate_random_string
from face_utils import load_face_data, save_face_img, upd_csv_video_faces, check_face_quality, check_orientation
from classes import FaceData

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
# cf hdbscan cluster
# chromadb

OUTPUT_FACES_FOLDER = "./faces_detected"
LANDMARK_MODEL_PATH = r"F:/PythonProjects/deepface/face_landmarker.task"
DETECTION_MODEL_PATH = r"F:/PythonProjects/deepface/blaze_face_short_range.tflite"
DETECTION_THRESHOLD = 0.5
BLUR_THRESHOLD = 1.8
MOTION_BLUR_THRESHOLD = 80.0


def remove_duplicate_faces(faces_detected: list[FaceData]) -> list[FaceData]:
    """
    Remove duplicate faces from a list of detected faces by comparing embeddings.
    
    Args:
        faces_detected (list[FaceData]): List of FaceData objects containing face embeddings
    Returns:
        list[FaceData]: List of unique FaceData objects after removing duplicates based on embedding similarity
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

    print(f"{RED}Removing {len(to_remove)} faces duplicated{RESET}")
    faces_detected = [face for index, face in enumerate(faces_detected) if index not in to_remove]

    return faces_detected


def add_embeddings(faces_detected: list[FaceData]) -> list[FaceData]:
    """
    Join embeddings to the faces detected by DeepFace in the FaceData objects list.
    
    Args:
        faces_detected (list): List of FaceData objects
    Returns:
        list: List of FaceData objects with embeddings added
    """
    faces_detected_with_embeddings = []
    for face_data in faces_detected:
        try:
            face_data.embedding = DeepFace.represent(face_data.face, model_name=MODEL, enforce_detection=False, detector_backend="skip")[0]["embedding"]
            faces_detected_with_embeddings.append(face_data)
        except Exception as e:
            print(f"Error DeepFace.represent: {e}")
    return faces_detected_with_embeddings


def discard_unqualified_faces(faces_detected: list[FaceData]) -> list[FaceData]:
    """
    Discard faces that do not meet the quality criteria such as blur and orientation.

    Args:
        faces_detected (list): List of FaceData objects
    Returns:
        list: List of FaceData objects
    """
    faces_detected_qualified = []
    base_options = python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1
                                        )
    detector = vision.FaceLandmarker.create_from_options(options)
    for face_data in faces_detected:
        if check_face_quality(face_data.face, BLUR_THRESHOLD, MOTION_BLUR_THRESHOLD) and check_orientation(face_data.face, detector):
            faces_detected_qualified.append(face_data)
    return faces_detected_qualified


def detect_faces_in_video(video : cv2.VideoCapture, rotation : int, frame_skip : int) -> list[FaceData]:
    """
    Process video frames to detect faces.
    
    Args:
        video (cv2.VideoCapture): OpenCV video capture object
        rotation (int): Video rotation angle in degrees (0, 90, 180, or 270)
        frame_skip (int): Number of frames to skip between processing. Higher values improve speed but may miss faces.
        
    Returns:
        list[FaceData]: List of detected faces with their associated data (face image, confidence score, width, height)
    """

    frame_count = 0
    faces_detected = []

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=DETECTION_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=DETECTION_THRESHOLD,
        min_suppression_threshold=0.2
        )
    
    with FaceDetector.create_from_options(options) as detector:
        while video.isOpened():
            ret, frame = video.read()
            frame_timestamp_ms = int(video.get(cv2.CAP_PROP_POS_MSEC))

            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = rotate_frame(frame, rotation)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            face_detector_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            if face_detector_result.detections:
              for detection in face_detector_result.detections:
                bounding_box = detection.bounding_box
                confidence = detection.categories[0].score
                face = frame[bounding_box.origin_y:bounding_box.origin_y + bounding_box.height, bounding_box.origin_x:bounding_box.origin_x + bounding_box.width]
                
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face_data = FaceData(
                        face=face,
                        confidence=confidence,
                        face_w=bounding_box.width,
                        face_h=bounding_box.height
                        )
                    faces_detected.append(face_data)
    return faces_detected


def verify_faces_detected(faces_detected: list[FaceData], old_faces_data: list) -> list[FaceData]:
    """
    Verify faces detected by comparing embeddings to old faces data.
    If the face is not a new face, confidences are compared to determine if the face is a more confident match than the old one.

    Args:
        faces_detected (list[FaceData]): List of FaceData objects containing face information
        old_faces_data (list): List of tuples containing (filename, embedding, confidence, size) for previously detected faces
    Returns:
        list[FaceData]: List of FaceData objects with updated match information
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
                print(f"Error DeepFace.verify: {e}")
    
    return faces_detected


def process_verified_faces(faces_detected: list[FaceData], old_faces_data: list) -> dict:
    """
    Process verified faces and update face data.
    If the face is a new face, it is added to the embeddings list.
    If the face is not a new face but has a better confidence, the old face is updated in the embeddings list.
    
    Args:
        faces_detected (list[FaceData]): List of FaceData objects containing face information
        old_faces_data (list): List of tuples containing (filename, embedding, confidence, size) for previously detected faces
    Returns:
        dict: Dictionary mapping filenames to tuples of (face_image, needs_saving), where needs_saving is True if the face
             is new or has better confidence than existing data
    """
    faces_in_processed_video = {}

    if len(faces_detected) == 0:
        return {}
    
    for face_data in faces_detected:
        if face_data.is_new and not face_data.filename:
            face_data.filename = f"{generate_random_string(16)}.jpg"
            old_faces_data.append((face_data.filename, face_data.embedding, face_data.confidence, face_data.face_h * face_data.face_w))
        elif face_data.is_better_match:
            old_faces_data[face_data.match_old_index] = (face_data.filename, face_data.embedding, face_data.confidence, face_data.face_h * face_data.face_w)
        faces_in_processed_video[face_data.filename] = [face_data.face, face_data.is_new or face_data.is_better_match]

    return faces_in_processed_video


def extract_faces(video_path, frame_skip=10, old_faces_data_path=None, csv_face_by_video_path=None):
    """
    Extract and process faces from a video file, saving unique faces and their embeddings.
    
    Args:
        video_path (str): Path to the video file
        frame_skip (int): Number of frames to skip between processing. Default is 10.
        old_faces_data_path (str): Path where face embeddings data are saved. Default is "face_data.npy".
        csv_face_by_video_path (str): Path to save face-video mapping CSV. Default is "face_by_video.csv".
    """

    if old_faces_data_path is None or old_faces_data_path == "":
        old_faces_data_path = "face_data.npy"
    if csv_face_by_video_path is None or csv_face_by_video_path == "":
        csv_face_by_video_path = "face_by_video.csv"

    video_name = os.path.basename(video_path)
    old_faces_data = load_face_data(old_faces_data_path)
    rotation = get_video_rotation(video_path)


    video = cv2.VideoCapture(video_path)
    output_folder = OUTPUT_FACES_FOLDER
    os.makedirs(output_folder, exist_ok=True)

    faces_detected = detect_faces_in_video(video, rotation, frame_skip)
    print(f"{GREEN}Faces detected : {len(faces_detected)}{RESET}")

    faces_detected_qualified = discard_unqualified_faces(faces_detected)
    print(f"{GREEN}Faces detected without unvalid faces : {len(faces_detected_qualified)}{RESET}")

    faces_qualified_embeddings = add_embeddings(faces_detected_qualified)

    faces_detected_unique = remove_duplicate_faces(faces_qualified_embeddings)
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
        video_path = "./videos/VID3.mp4"
        extract_faces(video_path, 10)
    except Exception as e:
        print(f"{RED}Error : {e}{RESET}")


if __name__ == "__main__":
    main()
