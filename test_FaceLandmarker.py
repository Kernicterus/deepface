from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from face_utils import extract_yaw_pitch_roll

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

model_path = r"F:/PythonProjects/deepface/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_facial_transformation_matrixes=True,
    num_faces=1
    )

cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
  while cap.isOpened():
    ret, frame = cap.read()
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ret:
      break
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    annotated_image = draw_landmarks_on_image(frame, face_landmarker_result)
    if face_landmarker_result.facial_transformation_matrixes:
      yaw, pitch, roll = extract_yaw_pitch_roll(face_landmarker_result.facial_transformation_matrixes[0])
      cv2.putText(annotated_image, f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("My webcam capture", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
