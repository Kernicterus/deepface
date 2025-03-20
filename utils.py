import cv2
import pandas as pd
import os
import random
import string

def generate_random_string(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))


def count_files_in_folder(folder_path, only_videos=False, only_images=False):
    """
    Compte le nombre de fichiers dans un dossier.
    
    Args:
        folder_path (str): Chemin du dossier à analyser
        only_videos (bool): Si True, compte uniquement les fichiers vidéo
        only_images (bool): Si True, compte uniquement les fichiers images
    Returns:
        int: Nombre de fichiers trouvés
    """
    count = 0
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    
    if not os.path.exists(folder_path):
        return 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if only_videos:
                if file.lower().endswith(video_extensions):
                    count += 1
            elif only_images:
                if file.lower().endswith(image_extensions):
                    count += 1
            else:
                count += 1
                
    return count


def rotate_frame(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame 


def get_video_rotation(video_path):
    """
    Opens a video and allows the user to navigate frame by frame using arrow keys,
    rotate the video with the 'R' key and validate with 'Enter'.
    
    Returns the rotation angle selected by the user (0, 90, 180 or 270 degrees).
    """
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0 
    rotation = 0 

    while True:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        
        if not ret:
            print("Error reading the video.")
            break

        frame = rotate_frame(frame, rotation)

        cv2.imshow("Adjust video - Left/Right arrows, R to rotate", frame)

        key = cv2.waitKeyEx(0)

        if key == ord('q'):
            rotation = 0
            break
        elif key == 0x250000:  
            frame_index = (frame_index - 1) % total_frames
        elif key == 0x270000:  
            frame_index = (frame_index + 1) % total_frames
        elif key == ord('r'): 
            rotation = (rotation + 90) % 360
        elif key == 13:
            break

    video.release()
    cv2.destroyAllWindows()

    return rotation


def get_video_paths(video_folder):
    """
    Extrait tous les chemins des fichiers vidéo dans le dossier spécifié.
    
    Args:
        video_folder (str): Chemin du dossier contenant les vidéos
        
    Returns:
        list: Liste des chemins complets des fichiers vidéo
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    
    video_paths = []
    
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                full_path = os.path.join(root, file)
                video_paths.append(full_path)
    
    return video_paths


def find_videos_by_id(file_id, csv_file):
    df = pd.read_csv(csv_file, skip_blank_lines=True)
    
    if 'video_path' not in df.columns:
        print("Erreur : La colonne 'video_path' est absente du fichier CSV.")
        return []

    id_columns = df.columns[1:]

    mask = df[id_columns].apply(lambda row: row.astype(str).str.contains(str(file_id), na=False), axis=1)

    matching_videos = df.loc[mask.any(axis=1), 'video_path'].tolist()
    
    return matching_videos

