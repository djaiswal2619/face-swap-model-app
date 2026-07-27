import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "lfw_funneled")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Model paths
SWAPPER_MODEL = os.path.join(MODELS_DIR, "inswapper_128.onnx")
CLASSIFIER_MODEL = os.path.join(MODELS_DIR, "face_classifier.pkl")
EMBEDDINGS_REGISTRY = os.path.join(MODELS_DIR, "embeddings_registry.npy")
METRICS_LOG = os.path.join(MODELS_DIR, "metrics_log.npy")

# Create directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)