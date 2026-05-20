import io
import os
import sys
import types
from pathlib import Path
from urllib.parse import parse_qs, urlparse, unquote
from urllib.request import urlopen, Request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


# ===== Standalone configuration (edit these paths directly) =====
REPO_ROOT = Path(__file__).resolve().parent
REPO_MODELS_DIR = REPO_ROOT / "models"

# IMPORTANT:
# Do NOT point MODELS_DIR to REPO_MODELS_DIR (./models).
# reactor_swapper has legacy migration logic that treats ./models as an old path
# and may try to move/remove it on startup.
# Use another directory, e.g. r"D:/Ai/AiTest/ReActorModels".
MODELS_DIR = Path(r"D:/Ai/AiTest/ReActor/_models")

# Optional: set a direct model file path (.onnx/.pth). If None, auto-discovery is used.
SWAP_MODEL_PATH = None
# ===============================================================


def _install_comfy_stubs() -> None:
    """Install minimal stubs so ReActor can run outside ComfyUI."""
    repo_root = Path(__file__).resolve().parent
    models_dir = str(MODELS_DIR.resolve())

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = models_dir
    folder_paths.supported_pt_extensions = {".pt", ".pth", ".onnx"}
    folder_paths.folder_names_and_paths = {}

    def add_model_folder_path(name: str, full_folder_path: str):
        paths, exts = folder_paths.folder_names_and_paths.get(name, ([], set()))
        if full_folder_path not in paths:
            paths.append(full_folder_path)
        folder_paths.folder_names_and_paths[name] = (paths, exts)

    folder_paths.add_model_folder_path = add_model_folder_path

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []  # mark as package for submodule imports

    model_management = types.ModuleType("comfy.model_management")
    utils = types.ModuleType("comfy.utils")

    def processing_interrupted() -> bool:
        return False

    def get_torch_device() -> str:
        return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    class ProgressBar:
        def __init__(self, total: int):
            self.total = max(int(total), 0)
            self.current = 0

        def update_absolute(self, value: int, total: int | None = None):
            if total is not None:
                self.total = max(int(total), 0)
            self.current = max(int(value), 0)

        def update(self, step: int = 1):
            self.current += int(step)

    def load_torch_file(path: str, safe_load: bool = True):
        import torch
        return torch.load(path, map_location="cpu")

    model_management.processing_interrupted = processing_interrupted
    model_management.get_torch_device = get_torch_device
    utils.ProgressBar = ProgressBar
    utils.load_torch_file = load_torch_file

    comfy.model_management = model_management
    comfy.utils = utils

    sys.modules["folder_paths"] = folder_paths
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.utils"] = utils



def _validate_model_paths() -> None:
    models_dir_resolved = MODELS_DIR.resolve()
    repo_models_resolved = REPO_MODELS_DIR.resolve()
    if models_dir_resolved == repo_models_resolved:
        raise RuntimeError(
            "Invalid MODELS_DIR: it points to './models' inside this repo. "
            "Set MODELS_DIR in run.py to another folder path (outside repo/models) "
            "to avoid legacy cleanup side effects in reactor_swapper."
        )


_validate_model_paths()


_install_comfy_stubs()

from scripts.reactor_swapper import swap_face  # noqa: E402


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _load_image(path_or_url: str) -> Image.Image:
    val = unquote(path_or_url)
    if _is_url(val):
        req = Request(val, headers={"User-Agent": "ReActor-Standalone/1.0"})
        with urlopen(req, timeout=60) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")

    img_path = Path(val)
    if not img_path.is_absolute():
        img_path = Path.cwd() / img_path
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    return Image.open(img_path).convert("RGB")


def _pick_swap_model() -> str:
    if SWAP_MODEL_PATH is not None:
        model_path = Path(SWAP_MODEL_PATH)
        if not model_path.is_absolute():
            model_path = (Path(__file__).resolve().parent / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Configured SWAP_MODEL_PATH does not exist: {model_path}")
        return str(model_path)

    models_root = Path(MODELS_DIR)
    candidates = []
    for subdir in ("hyperswap", "reswapper", "insightface"):
        root = models_root / subdir
        if not root.exists():
            continue
        candidates.extend(sorted(root.glob("*.onnx")))
        candidates.extend(sorted(root.glob("*.pth")))
    if not candidates:
        raise FileNotFoundError(
            f"No swap model found under {models_root}. "
            "Expected at least one .onnx or .pth inside models/hyperswap, models/reswapper or models/insightface"
        )
    return str(candidates[0])


class SwapHandler(BaseHTTPRequestHandler):
    model_path = _pick_swap_model()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/swap":
            self.send_error(404, "Use /swap")
            return

        params = parse_qs(parsed.query)
        source_url = params.get("source_url", [None])[0]
        target_url = params.get("target_url", [None])[0]
        if not source_url or not target_url:
            self.send_error(400, "source_url and target_url are required")
            return

        try:
            source_img = _load_image(source_url)
            target_img = _load_image(target_url)
            swapped_img, _, _ = swap_face(
                source_img=source_img,
                target_img=target_img,
                model=self.model_path,
                source_faces_index=[0],
                faces_index=[0],
                gender_source=0,
                gender_target=0,
                faces_order=["large-small", "large-small"],
                face_boost_enabled=False,
            )

            output = io.BytesIO()
            swapped_img.save(output, format="JPEG", quality=95)
            body = output.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            error = str(exc).encode("utf-8", errors="ignore")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(error)))
            self.end_headers()
            self.wfile.write(error)


def main() -> None:
    host = os.environ.get("REACTOR_HOST", "0.0.0.0")
    port = int(os.environ.get("REACTOR_PORT", "8004"))
    server = ThreadingHTTPServer((host, port), SwapHandler)
    print(f"ReActor standalone API is running on http://{host}:{port}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Model file: {SwapHandler.model_path}")
    print("Example: /swap?source_url=./source.jpg&target_url=https%3A%2F%2Fexample.com%2Ftarget.jpg")
    server.serve_forever()


if __name__ == "__main__":
    main()
