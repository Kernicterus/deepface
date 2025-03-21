import os
import numpy as np
import cv2
import pandas as pd
import mediapipe as mp

YELLOW = "\033[33m"
RED = "\033[31m"
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


def check_face_quality(img_array, blur_threshold=2.0, motion_blur_threshold=80.0):
    blur_score = blur_face_score(img_array)
    motion_blur = motion_blur_score(img_array)
    print(f"Blur score: {blur_score}, Motion blur: {motion_blur}")
    if blur_score > 10 :
        return True
    elif blur_score < blur_threshold or motion_blur < motion_blur_threshold:
        print("Face too blurry")
        return False

    return True


def extract_yaw_pitch_roll(transformation_matrix):
    R = transformation_matrix[:3, :3]

    roll = np.arctan2(R[1, 0], R[0, 0]) * (180 / np.pi)
    yaw = np.arcsin(-R[2, 0]) * (180 / np.pi)
    pitch = np.arctan2(R[2, 1], R[2, 2]) * (180 / np.pi)

    return yaw, pitch, roll


def check_orientation(face, detector):
    print(f"{YELLOW}Checking face orientation{RESET}")
    face = np.ascontiguousarray(face, dtype=np.uint8)
    face_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=face)
    # face_mp =face
    detection_result = detector.detect(face_mp)
    if detection_result.facial_transformation_matrixes:
        yaw, pitch, roll = extract_yaw_pitch_roll(detection_result.facial_transformation_matrixes[0])
        print(f"yaw: {yaw}, pitch: {pitch}, roll: {roll}")
        if yaw > 25 or yaw < -25 or pitch > 25 or pitch < -25:
            print(f"{RED}Face orientation is not good{RESET}")
            cv2.imshow("Face", face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return False
        return True
    else:
        print(f"{RED}Orientation evaluation not possible{RESET}")
        cv2.imshow("Face", face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False

