from src.data_loader import load_ct_scan
from src.preprocessing import preprocess_scan
from src.rag.report_generator import create_report
from src import infer as infer_module
from src.train import train as train_fn
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    scan = load_ct_scan("data/samples/sample_scan.mhd")
except Exception as e:
    print("Failed to load CT scan, generating synthetic data:", e)
    import numpy as np
    scan = np.random.randint(-1200, 700, size=(10, 512, 512)).astype(np.int16)

processed = preprocess_scan(scan)

# Train only if PyTorch is available
if infer_module.is_torch_available():
    try:
        train_fn(processed[:10])
    except Exception as e:
        print("Training failed:", e)
else:
    print("PyTorch not available. Skipping training.")

# Prepare a single slice as a PIL image for inference
arr = (processed[0] * 255).astype(np.uint8)
img = Image.fromarray(arr)

model = None
if infer_module.is_torch_available():
    try:
        model = infer_module.load_model("models/baseline_model.pth", device='cpu')
    except Exception as e:
        print("Failed to load model, running in demo mode:", e)

confidence = infer_module.predict(model, img, device='cpu')

create_report(confidence)

