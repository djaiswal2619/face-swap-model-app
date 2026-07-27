import cv2
from core.swapper import FaceSwapper


class VideoProcessor:
    def __init__(self, swapper: FaceSwapper):
        self.swapper = swapper

    def process(self, video_path: str, output_path: str, source_img_path: str):
        """Process video and swap faces frame by frame."""
        source_img = cv2.imread(source_img_path)
        if source_img is None:
            raise ValueError("Could not read source image")

        source_face = self.swapper.get_face(source_img)
        if source_face is None:
            raise ValueError("No face found in source image")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            target_face = self.swapper.get_face(frame)
            if target_face:
                frame = self.swapper.swap_faces(source_img, frame)

            out.write(frame)

        cap.release()
        out.release()
        print(f"Video saved: {output_path}")