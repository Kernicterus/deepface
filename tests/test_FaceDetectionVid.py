import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from typing import Tuple, Union
import math
import numpy as np
from face_utils import getYawPitchRoll

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

    
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    yaw, pitch, roll = getYawPitchRoll(detection)
    cv2.putText(annotated_image, f"yaw: {yaw:.2f}, pitch: {pitch:.2f}, roll: {roll:.2f}", (text_location[0], text_location[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

model_path = r"F:/PythonProjects/deepface/blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.5,
    min_suppression_threshold=0.2
    )

cap = cv2.VideoCapture(r"F:/PythonProjects/deepface/vid1/VID3.mp4")
frame_count = 0
with FaceDetector.create_from_options(options) as detector:
  while cap.isOpened():
    ret, frame = cap.read()
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if not ret:
      break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    i = 0
    frame_count += 1
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    face_detector_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    if face_detector_result.detections:
      for detection in face_detector_result.detections:
        bounding_box = detection.bounding_box
        face = frame[bounding_box.origin_y:bounding_box.origin_y + bounding_box.height, bounding_box.origin_x:bounding_box.origin_x + bounding_box.width]
        cv2.rectangle(frame, (detection.bounding_box.origin_x, detection.bounding_box.origin_y), (detection.bounding_box.origin_x + detection.bounding_box.width, detection.bounding_box.origin_y + detection.bounding_box.height), (0, 0, 255), 2)
        cv2.imwrite(f"F:/PythonProjects/deepface/debug/frame{frame_count}_detection_{i}.jpg", face)
        i += 1
    # annotated_image = visualize(frame, face_detector_result)
    # cv2.imshow("My webcam capture", annotated_image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
    
cap.release()
cv2.destroyAllWindows()
