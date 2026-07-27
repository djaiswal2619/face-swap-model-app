import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


class FaceSwapper:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join("models", "inswapper_128.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.swapper = insightface.model_zoo.get_model(model_path)
        print("Face swapper initialized successfully")

    def get_face(self, img: np.ndarray):
        faces = self.app.get(img)
        if not faces:
            return None
        # Return largest face (main subject)
        return max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    def swap_faces(self, source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        source_face = self.get_face(source_img)
        target_face = self.get_face(target_img)

        if source_face is None or target_face is None:
            raise ValueError("Face detection failed")

        return self.swapper.get(target_img, target_face, source_face, paste_back=True)