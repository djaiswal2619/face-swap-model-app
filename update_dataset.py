import os
import cv2
import numpy as np
import config
from core.swapper import FaceSwapper


def build_embeddings_registry(dataset_path: str = config.DATASET_DIR,
                              registry_path: str = config.EMBEDDINGS_REGISTRY):
    """
    Scan dataset folders and extract face embeddings.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    swapper = FaceSwapper()

    embeddings = []
    labels = []
    skipped = 0

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        person_count = 0

        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            face = swapper.get_face(img)
            if face is None:
                skipped += 1
                continue

            embeddings.append(face.embedding)
            labels.append(person_name)
            person_count += 1

        if person_count > 0:
            print(f"  {person_name}: {person_count} images")

    if len(embeddings) == 0:
        raise ValueError("No faces found in dataset")

    np.save(registry_path, {
        'embeddings': np.array(embeddings),
        'labels': np.array(labels)
    })

    print(f"\nRegistry saved: {registry_path}")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Unique identities: {len(np.unique(labels))}")
    print(f"Skipped images: {skipped}")


if __name__ == "__main__":
    build_embeddings_registry()