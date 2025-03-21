import cv2
import pandas as pd
import os
import random
import string
from pymediainfo import MediaInfo

def get_video_rotation(video_path):
    """
    Get the rotation of a video by using pymediainfo to parse the video metadata.

    Args:
        video_path (str): Path to the video
        
    Returns:
        int: Rotation to apply to the video to be displayed correctly
    """
    media_info = MediaInfo.parse(video_path)
    for track in media_info.tracks:
        if track.track_type == "Video":
            try:
                return float(track.rotation)
            except:
                return None
    return None


def rotate_frame(frame, rotation):
    if rotation > 45 and rotation < 135:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation > 225 and rotation < 315:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation > 135 and rotation < 225:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame 


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





# def get_video_rotation(video_path):
#     """
#     Opens a video and allows the user to navigate frame by frame using arrow keys,
#     rotate the video with the 'R' key and validate with 'Enter'.
    
#     Returns the rotation angle selected by the user (0, 90, 180 or 270 degrees).
#     """
#     video = cv2.VideoCapture(video_path)
#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_index = 0 
#     rotation = 0 

#     while True:
#         video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
#         ret, frame = video.read()
        
#         if not ret:
#             print("Error reading the video.")
#             break

#         frame = rotate_frame(frame, rotation)

#         cv2.imshow("Adjust video - Left/Right arrows, R to rotate", frame)

#         key = cv2.waitKeyEx(0)

#         if key == ord('q'):
#             rotation = 0
#             break
#         elif key == 0x250000:  
#             frame_index = (frame_index - 1) % total_frames
#         elif key == 0x270000:  
#             frame_index = (frame_index + 1) % total_frames
#         elif key == ord('r'): 
#             rotation = (rotation + 90) % 360
#         elif key == 13:
#             break

#     video.release()
#     cv2.destroyAllWindows()

#     return rotation


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


def extract_video_metadata(video : cv2.VideoCapture):
    """
    Extracts and displays all metadata from a video.
    
    Args:
        video_path (str): Path to the video
    """
    
    if not video.isOpened():
        print(f"Error: Unable to open video {video}")
        return
    
    print("\n=== Video Metadata ===")

    
    print("\nVideo Properties:")
    print(f"Width: {int(video.get(cv2.CAP_PROP_FRAME_WIDTH))} pixels")
    print(f"Height: {int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))} pixels")
    print(f"Total frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"FPS: {video.get(cv2.CAP_PROP_FPS):.2f}")
    print(f"Duration: {int(video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS))} seconds")
    
    print("\nCodec and format:")
    print(f"Codec: {int(video.get(cv2.CAP_PROP_FOURCC))}")
    print(f"Format: {chr(int(video.get(cv2.CAP_PROP_FOURCC)) & 0xFF) + chr((int(video.get(cv2.CAP_PROP_FOURCC)) >> 8) & 0xFF) + chr((int(video.get(cv2.CAP_PROP_FOURCC)) >> 16) & 0xFF) + chr((int(video.get(cv2.CAP_PROP_FOURCC)) >> 24) & 0xFF)}")
    
    print("\nOther properties:")
    print(f"Brightness: {video.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast: {video.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"Saturation: {video.get(cv2.CAP_PROP_SATURATION)}")
    print(f"Hue: {video.get(cv2.CAP_PROP_HUE)}")
    print(f"Gain: {video.get(cv2.CAP_PROP_GAIN)}")
    print(f"Exposure: {video.get(cv2.CAP_PROP_EXPOSURE)}")
    
    print("\nEXIF Properties:")
    try:
        exif_data = {}
        for i in range(100):
            value = video.get(i)
            if value != -1:
                exif_data[i] = value
        
        if exif_data:
            print("EXIF metadata found:")
            
            # Dimensions
            if 3 in exif_data and 4 in exif_data:
                print(f"Resolution: {int(exif_data[3])}x{int(exif_data[4])} pixels")
            
            # FPS and duration
            if 5 in exif_data:
                print(f"FPS: {exif_data[5]:.2f}")
            if 7 in exif_data:
                print(f"Duration: {int(exif_data[7])} seconds")
            
            # Dates
            if 6 in exif_data:
                print(f"Creation date: {exif_data[6]}")
            if 42 in exif_data:
                print(f"Year: {int(exif_data[42])}")
            if 46 in exif_data:
                print(f"Modification date: {exif_data[46]}")
            
            # Orientation
            if 48 in exif_data:
                print(f"EXIF rotation: {int(exif_data[48])} degrees")
            
            # Quality
            if 68 in exif_data and 69 in exif_data and 70 in exif_data:
                print(f"Quality: {exif_data[68]} (Level: {exif_data[69]}, Depth: {exif_data[70]})")
            
            # File size
            if 47 in exif_data:
                print(f"File size: {int(exif_data[47])} bytes")
        else:
            print("No EXIF metadata found")
    except Exception as e:
        print(f"Error extracting EXIF metadata: {e}")

    video.release()
    print("\n==============================\n")

