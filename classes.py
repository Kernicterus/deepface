from dataclasses import dataclass
import numpy as np

@dataclass  
class FaceData:
    """
    Represents a detected face with its associated data.
    """
    face: np.ndarray
    confidence: float
    face_w: int
    face_h: int
    embedding: np.ndarray = None
    is_new: bool = True
    match_old_index: int = None
    is_better_match: bool = False
    filename: str = None
