import hashlib
import io
import os
import sys
import threading
import time
import types
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor

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

# Device mode for Comfy stub: "cpu" or "gpu".
# - "gpu": forces torch device to "cuda".
# - "cpu": forces torch device to "cpu".
# - You can also set REACTOR_DEVICE env var to override at runtime.
DEVICE_MODE = "gpu"

# Concurrency limit for simultaneous /swap processing.
# Even if 20+ requests arrive together, only this many are processed concurrently.
MAX_CONCURRENT_REQUESTS = 4

# Cache root folder for both downloaded inputs and processed outputs.
# Structure:
# - CACHE_DIR/sources/<hash_of_source_url_or_path>.img
# - CACHE_DIR/targets/<hash_of_target_url_or_path>.img
# - CACHE_DIR/results/<hash_of_source+target+params>.jpg
CACHE_DIR = REPO_ROOT / "cache"
SOURCES_CACHE_DIR = CACHE_DIR / "sources"
TARGETS_CACHE_DIR = CACHE_DIR / "targets"
RESULTS_CACHE_DIR = CACHE_DIR / "results"
# ===============================================================

SWAP_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
RESULT_LOCK = threading.Lock()


def _ensure_cache_dirs() -> None:
    SOURCES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TARGETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_sha256_hex(image: Image.Image) -> str:
    normalized = image.convert("RGB")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _install_comfy_stubs() -> None:
    """Install minimal stubs so ReActor can run outside ComfyUI."""
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
    comfy.__path__ = []

    model_management = types.ModuleType("comfy.model_management")
    utils = types.ModuleType("comfy.utils")

    def processing_interrupted() -> bool:
        return False

    def get_torch_device() -> str:
        requested = os.environ.get("REACTOR_DEVICE", DEVICE_MODE).lower()
        if requested == "gpu":
            return "cuda"
        return "cpu"

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
_ensure_cache_dirs()
_install_comfy_stubs()

from scripts.reactor_swapper import swap_face  # noqa: E402


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _try_decode_base64_image(value: str) -> Image.Image | None:
    candidate = value.strip()
    marker_idx = candidate.find("base64,")
    if marker_idx == -1:
        return None

    import base64
    import io

    payload = candidate[marker_idx + len("base64,") :]
    normalized_payload = "".join(payload.split())
    try:
        decoded = base64.b64decode(normalized_payload)
    except Exception:
        return None

    try:
        return Image.open(io.BytesIO(decoded)).convert("RGB")
    except Exception:
        return None


def _load_image(path_or_url: str, cache_subdir: Path) -> Image.Image:
    """
    Load the *current* image for a source/target reference.

    Cache policy:
    - Never trust URL/path identity as image identity.
    - Always resolve fresh bytes first, then cache by content hash.
    - This guarantees we only reuse cache when source/target content is truly unchanged.
    """
    cache_subdir.mkdir(parents=True, exist_ok=True)

    decoded_inline_image = _try_decode_base64_image(path_or_url)
    if decoded_inline_image is not None:
        content_hash = _image_sha256_hex(decoded_inline_image)
        cache_file = cache_subdir / f"{content_hash}.png"
        if cache_file.exists():
            return Image.open(cache_file).convert("RGB")
        decoded_inline_image.save(cache_file, format="PNG")
        return decoded_inline_image

    # IMPORTANT:
    # - query parsing already decodes URL parameters once.
    # - some CDNs include encoded characters inside path segments (e.g. %20).
    # If we unquote() a remote URL again, %20 turns into a literal space and
    # urllib raises "URL can't contain control characters".
    # So for HTTP(S), keep the URL as-is and do not unquote it again.
    if _is_url(path_or_url):
        req = Request(path_or_url, headers={"User-Agent": "ReActor-Standalone/1.0"})
        with urlopen(req, timeout=60) as response:
            data = response.read()

        image = Image.open(io.BytesIO(data)).convert("RGB")
        content_hash = _image_sha256_hex(image)
        cache_file = cache_subdir / f"{content_hash}.png"
        if not cache_file.exists():
            image.save(cache_file, format="PNG")
        return image

    val = unquote(path_or_url)
    img_path = Path(val)
    if not img_path.is_absolute():
        img_path = Path.cwd() / img_path
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    local_img = Image.open(img_path).convert("RGB")
    content_hash = _image_sha256_hex(local_img)
    cache_file = cache_subdir / f"{content_hash}.png"
    if cache_file.exists():
        return Image.open(cache_file).convert("RGB")
    local_img.save(cache_file, format="PNG")
    return local_img


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


def _build_swap_options(params: dict[str, list[str]]) -> dict:
    """Map URL query parameters to swap_face options.

    Examples:
    - source_faces_index=0,1
    - faces_index=0
    - gender_source=0
    - gender_target=0
    - faces_order=large-small,large-small
    - face_boost_enabled=true
    """

    def _bool(name: str, default: bool) -> bool:
        raw = params.get(name, [str(default)])[0].strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _int(name: str, default: int) -> int:
        return int(params.get(name, [str(default)])[0])

    def _int_list(name: str, default: list[int]) -> list[int]:
        raw = params.get(name, [None])[0]
        if raw is None or raw.strip() == "":
            return default
        return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]

    def _str_list(name: str, default: list[str]) -> list[str]:
        raw = params.get(name, [None])[0]
        if raw is None or raw.strip() == "":
            return default
        return [x.strip() for x in raw.split(",") if x.strip() != ""]

    return {
        "source_faces_index": _int_list("source_faces_index", [0]),
        "faces_index": _int_list("faces_index", [0]),
        "gender_source": _int("gender_source", 0),
        "gender_target": _int("gender_target", 0),
        "faces_order": _str_list("faces_order", ["large-small", "large-small"]),
        "face_boost_enabled": _bool("face_boost_enabled", False),
    }


class SwapHandler(BaseHTTPRequestHandler):
    model_path = _pick_swap_model()

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        request_start = time.perf_counter()
        parsed = urlparse(self.path)
        if parsed.path != "/swap":
            self.send_error(404, "Use /swap")
            self._log_request_timing(request_start, "invalid_path")
            return

        params = parse_qs(parsed.query)
        source_url = params.get("source_url", [None])[0]
        target_url = params.get("target_url", [None])[0]
        if not source_url or not target_url:
            self.send_error(400, "source_url and target_url are required")
            self._log_request_timing(request_start, "missing_required_params")
            return

        try:
            swap_options = _build_swap_options(params)
            source_img = _load_image(source_url, SOURCES_CACHE_DIR)
            target_img = _load_image(target_url, TARGETS_CACHE_DIR)
            source_hash = _image_sha256_hex(source_img)
            target_hash = _image_sha256_hex(target_img)
            source_result_dir = RESULTS_CACHE_DIR / source_hash
            source_result_dir.mkdir(parents=True, exist_ok=True)
            cache_key = _sha256_hex(
                f"source={source_hash}|target={target_hash}|opts={repr(sorted(swap_options.items()))}|model={self.model_path}"
            )
            result_path = source_result_dir / f"{cache_key}.jpg"

            def _run_swap() -> bytes:
                swapped_img, _, _ = swap_face(
                    source_img=source_img,
                    target_img=target_img,
                    model=self.model_path,
                    **swap_options,
                )
                output = io.BytesIO()
                swapped_img.save(output, format="JPEG", quality=95)
                body_inner = output.getvalue()
                with RESULT_LOCK:
                    if not result_path.exists():
                        result_path.write_bytes(body_inner)
                return body_inner

            with RESULT_LOCK:
                if result_path.exists():
                    body = result_path.read_bytes()
                    self._send_jpeg(body)
                    self._log_request_timing(request_start, "cache_hit")
                    return

            body = SWAP_EXECUTOR.submit(_run_swap).result()
            self._send_jpeg(body)
            self._log_request_timing(request_start, "processed")
        except Exception as exc:
            error = str(exc).encode("utf-8", errors="ignore")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(error)))
            self.end_headers()
            self.wfile.write(error)
            self._log_request_timing(request_start, "error")

    def _send_jpeg(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_request_timing(self, start: float, outcome: str) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"[swap] outcome={outcome} elapsed_ms={elapsed_ms:.2f} path={self.path}")


def main() -> None:
    host = os.environ.get("REACTOR_HOST", "0.0.0.0")
    port = int(os.environ.get("REACTOR_PORT", "8004"))
    server = ThreadingHTTPServer((host, port), SwapHandler)
    print(f"ReActor standalone API is running on http://{host}:{port}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Model file: {SwapHandler.model_path}")
    print(f"Device mode: {os.environ.get('REACTOR_DEVICE', DEVICE_MODE)}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Cache dir: {CACHE_DIR}")
    print("Example: /swap?source_url=./source.jpg&target_url=https%3A%2F%2Fexample.com%2Ftarget.jpg")
    print("Example with params: &source_faces_index=0,1&faces_index=0&face_boost_enabled=true")
    server.serve_forever()


if __name__ == "__main__":
    main()
