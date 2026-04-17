"""
Web Application - Enhanced Steel Diagnostic Tool
==================================================
FastAPI server with dual-mode analysis:
  - Microstructure mode: classify UHCS micrographs + metallurgical report + defect flags
  - Surface mode: classify NEU surface defects + root cause + remedies

Usage:
    python app.py
    -> Open http://localhost:8000 in your browser
"""

import os
import io
import base64
import numpy as np
import torch
from PIL import Image
import cv2
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from step4_model import create_model, get_transforms, GradCAM
from knowledge_base import UHCS_TO_KB_MAP, get_knowledge
from defect_knowledge_base import NEU_TO_KB_MAP, get_surface_defect, get_micro_defect


# -- Configuration ---------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
MICRO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_microstructure_model.pth")
SURFACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_surface_defect_model.pth")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
IMAGE_SIZE = 224


# -- App Setup -------------------------------------------------------------------

app = FastAPI(title="Enhanced Steel Diagnostic Tool", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


# -- Model Managers (lazy singletons) -------------------------------------------

class BaseModelManager:
    """Base class for lazy-loaded model singletons."""

    def __init__(self):
        self.model = None
        self.class_names = None
        self.grad_cam = None
        self.transform = None
        self.device = None
        self.kb_map = {}

    def _load_model(self, model_path, kb_mapping):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading model from {os.path.basename(model_path)} on {self.device}...")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint.get("class_names") or checkpoint.get("classes")
        num_classes = len(self.class_names)

        self.model = create_model(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.grad_cam = GradCAM(self.model, self.model.layer4)
        self.transform = get_transforms(is_training=False)

        for folder_name in self.class_names:
            self.kb_map[folder_name] = kb_mapping.get(folder_name, folder_name)

        print(f"  Model loaded! Classes: {self.class_names}")

    def _run_prediction(self, pil_image):
        """Shared prediction logic: classify + Grad-CAM."""
        img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Classification
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        kb_name = self.kb_map.get(pred_class, pred_class)
        all_probs = {self.kb_map.get(c, c): float(probs[i])
                     for i, c in enumerate(self.class_names)}

        # Grad-CAM
        img_tensor_grad = self.transform(img).unsqueeze(0).to(self.device)
        img_tensor_grad.requires_grad_(True)
        self.model.train()
        heatmap = self.grad_cam.generate(img_tensor_grad, pred_idx)
        self.model.eval()

        # Create overlay
        img_array = np.array(img)
        heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (img_array * 0.5 + heatmap_colored * 0.5).astype(np.uint8)

        def to_base64(arr):
            pil = Image.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "class_name": pred_class,
            "kb_name": kb_name,
            "confidence": confidence,
            "all_probs": all_probs,
            "predicted_index": pred_idx,
            "original_b64": to_base64(img_array),
            "heatmap_b64": to_base64(heatmap_colored),
            "overlay_b64": to_base64(overlay),
        }


class MicroModelManager(BaseModelManager):
    """Singleton for microstructure classification model."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    def load(self):
        self._load_model(MICRO_MODEL_PATH, UHCS_TO_KB_MAP)

    def predict(self, pil_image):
        result = self._run_prediction(pil_image)
        # Add microstructure knowledge
        result["knowledge"] = get_knowledge(result["kb_name"])
        # Check for microstructural defect flag
        defect_info = get_micro_defect(result["kb_name"])
        if defect_info:
            result["is_defect"] = True
            result["defect_info"] = defect_info
        else:
            result["is_defect"] = False
            result["defect_info"] = None
        return result


class SurfaceModelManager(BaseModelManager):
    """Singleton for surface defect classification model."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    def load(self):
        self._load_model(SURFACE_MODEL_PATH, NEU_TO_KB_MAP)

    def predict(self, pil_image):
        result = self._run_prediction(pil_image)
        # Add surface defect knowledge
        result["knowledge"] = get_surface_defect(result["kb_name"])
        result["is_defect"] = True  # All surface detections are defects
        return result


# -- Routes ----------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main upload page."""
    try:
        return templates.TemplateResponse(request=request, name="index.html")
    except Exception as e:
        import traceback
        return HTMLResponse(content=f"<pre>Error loading index.html:\n{str(e)}\n\n{traceback.format_exc()}</pre>", status_code=500)


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    mode: str = Form(default="micro"),
):
    """Analyze an uploaded image in the specified mode."""
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        if mode == "surface":
            manager = SurfaceModelManager.get()
        else:
            manager = MicroModelManager.get()

        result = manager.predict(pil_image)
        result["mode"] = mode
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        return JSONResponse(
            content={"error": f"Model not found. Train the model first: {str(e)}"},
            status_code=404
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "micro_model_loaded": MicroModelManager._instance is not None,
        "surface_model_loaded": SurfaceModelManager._instance is not None,
    }


# -- Main ------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Enhanced Steel Diagnostic Tool ===")
    print("Loading models...")

    # Try to pre-load microstructure model (optional, will lazy-load on first request)
    try:
        MicroModelManager.get()
    except Exception as e:
        print(f"  [!] Microstructure model not available: {e}")
        print(f"      Train with: python step5_train.py")

    # Try to pre-load surface defect model
    try:
        SurfaceModelManager.get()
    except Exception as e:
        print(f"  [!] Surface defect model not available: {e}")
        print(f"      Train with: python step5_train.py --dataset neu")

    port = int(os.environ.get("PORT", 8000))
    print(f"\nStarting server at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
