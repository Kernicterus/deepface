# DeepFace Video Analysis

This project is a video analysis tool that uses deep learning to detect and analyze faces in videos. It provides a user-friendly interface for processing videos and extracting face data.

## Features

- Face detection in videos
- Face orientation analysis (yaw, pitch, roll)
- Video metadata extraction
- Automatic video rotation detection
- Face quality assessment
- Face tracking across video frames
- CSV export of face detection results

## Requirements

- Python 3.8 or higher
- OpenCV
- MediaPipe
- DeepFace
- Tkinter (for the UI)
- pymediainfo
- pandas
- numpy

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Launch the program by running:
```bash
python UI.py
```

2. The main interface will allow you to:
   - Select video files or folders to process
   - Configure processing parameters
   - Export results
   - Search for associations between face images and videos where they appear via the Search tab

## Project Structure

- `UI.py`: Main user interface
- `utils.py`: Utility functions for video processing
- `face_utils.py`: Face detection and analysis functions
- `requirements.txt`: List of required Python packages

## Output

The program generates:
- Processed face images in the `faces_detected` folder
- CSV file containing face detection results
- NPY file congaining faces embeddings

## Notes

- The program automatically detects and corrects video orientation
- Processing time can be reduced by increasing the frame skip value in the UI, but this may result in missed faces
- Processing time depends on video length and resolution
- To ensure that a face previously detected in an old video is recognized as the same face in a new video, make sure to select both the CSV and NPY files in the UI before processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
