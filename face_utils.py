import os
import numpy as np
import cv2
import pandas as pd
from mtcnn import MTCNN
# import mediapipe as mp

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

YELLOW = "\033[33m"
RESET = "\033[0m"

def load_face_data(face_data_path):
    """
    Load existing face data from file or create empty list if none exists.
    
    Args:
        face_data_path (str): Path to face data file
        
    Returns:
        list: Face data containing filenames, embeddings, confidence scores and sizes
    """
    if os.path.exists(face_data_path):
        return np.load(face_data_path, allow_pickle=True).tolist()
    return []


def save_face_img(faces_in_video_processed, output_folder):
    if faces_in_video_processed is None or len(faces_in_video_processed) == 0:
        return
    for face_filename, (face, to_save) in faces_in_video_processed.items():
        if to_save:
            face_path = os.path.join(output_folder, face_filename)
            cv2.imwrite(face_path, face)


def upd_csv_video_faces(faces_in_video_processed : dict, video_name, csv_path):
    """
    Updates or adds a row in a CSV file with the video path and associated faces.
    
    Args:
        faces_in_video_processed (dict): Dictionary of faces to record in the following columns
        video_path (str): Path of the video to record
        csv_path (str): Path to the CSV file
    """
    if faces_in_video_processed is None or len(faces_in_video_processed) == 0:
        return
    
    video_path = video_name
    faces_list = [key for key in faces_in_video_processed.keys()]

    if not os.path.exists(csv_path):
        headers = ['video_path'] + [f'face{i+1}' for i in range(len(faces_list))]
        df = pd.DataFrame(columns=headers)
    else:
        df = pd.read_csv(csv_path)

        current_cols = len(df.columns)
        needed_cols = len(faces_list) + 1 
        if current_cols < needed_cols:
            for i in range(current_cols - 1, needed_cols - 1):
                df[f'face{i+1}'] = None
        elif current_cols > needed_cols:
            faces_list.extend([None] * (current_cols - needed_cols))
    
    new_data = {'video_path': video_path}
    for i, face in enumerate(faces_list):
        new_data[f'face{i+1}'] = face
    
    if video_path in df['video_path'].values:
        df.loc[df['video_path'] == video_path, list(new_data.keys())] = list(new_data.values())
    else:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    
    df.to_csv(csv_path, index=False)


def blur_face_score(npy_face: np.ndarray) -> float:
    if len(npy_face.shape) == 3 and npy_face.shape[2] == 3:
        npy_face = cv2.cvtColor(npy_face, cv2.COLOR_RGB2GRAY)

    laplacian = cv2.Laplacian(npy_face, cv2.CV_64F)
    variance = laplacian.var()

    max_blur = 300.0
    score = min(100, max(0, (variance / max_blur) * 100))

    return round(score, 2)


def motion_blur_score(image : np.ndarray) -> float:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    var_x = np.var(sobel_x)
    var_y = np.var(sobel_y)

    motion_blur = abs(var_x - var_y) / (var_x + var_y + 1e-5)

    score = (1 - motion_blur) * 100
    return round(score, 2)


def align_face(img_array, detector):
    if img_array.shape[-1] == 3 and np.mean(img_array[:, :, 0]) < np.mean(img_array[:, :, 2]):
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = detector.detect_faces(img_array)

    keypoints = results[0]['keypoints']
    
    if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
        print("Impossible to align face, missing key points")
        return img_array

    left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']

    left_eye = (int(left_eye[0]), int(left_eye[1]))
    right_eye = (int(right_eye[0]), int(right_eye[1]))

    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]

    if abs(dx) < 1:
        print("Distance between eyes too small, alignment ignored")
        return img_array

    angle = np.degrees(np.arctan2(dy, dx))

    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    if not isinstance(center[0], int) or not isinstance(center[1], int):
        print(f"Error: center has an invalid format {center}")
        return img_array

    M = cv2.getRotationMatrix2D(center, angle, 1)
    aligned_face = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))

    # x, y, w, h = results[0]['box']
    # x, y = max(0, x), max(0, y)
    # aligned_face = aligned_face[y:y + h, x:x + w]

    return aligned_face

def check_face_quality(img_array, detector):
    blur_score = blur_face_score(img_array)
    motion_blur = motion_blur_score(img_array)

    if blur_score < 2 or motion_blur < 80:
        # print("Face too blurry")
        return False

    if img_array.shape[-1] == 3 and np.mean(img_array[:, :, 0]) < np.mean(img_array[:, :, 2]):
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = detector.detect_faces(img_array)
    if len(results) == 0:
        # print("No face detected")
        return False
    return True


# def estimate_rotation_angle(img_array):
#     # Convertir l'image en RGB
#     image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

#     # Détecter les points de repère faciaux
#     results = face_mesh.process(image_rgb)

#     if results.multi_face_landmarks:
#         landmarks = results.multi_face_landmarks[0].landmark

#         # Extraire les coordonnées des points de repère clés
#         left_eye = np.array([landmarks[33].x, landmarks[33].y])
#         right_eye = np.array([landmarks[263].x, landmarks[263].y])

#         # Calculer l'angle de rotation
#         dY = right_eye[1] - left_eye[1]
#         dX = right_eye[0] - left_eye[0]
#         angle = np.degrees(np.arctan2(dY, dX))

#         return angle
#     else:
#         return "Aucun visage détecté"

