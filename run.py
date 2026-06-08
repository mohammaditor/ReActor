import hashlib
import io
import os
import sys
import threading
import time
import logging
import uuid
import types
import ssl
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import requests
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
MODELS_DIR = REPO_ROOT / "_models"

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

# If True, video frame processing is distributed across all Uvicorn workers
# by sending internal HTTP requests to localhost.
DISTRIBUTED_VIDEO_PROCESSING = False

# Cache root folder for both downloaded inputs and processed outputs.
# Structure:
# - CACHE_DIR/sources/<hash_of_source_url_or_path>.img
# - CACHE_DIR/targets/<hash_of_target_url_or_path>.img
# - CACHE_DIR/results/<hash_of_source+target+params>.jpg
CACHE_DIR = REPO_ROOT / "cache"
SOURCES_CACHE_DIR = CACHE_DIR / "sources"
TARGETS_CACHE_DIR = CACHE_DIR / "targets"
RESULTS_CACHE_DIR = CACHE_DIR / "results"
FACES_CACHE_DIR = CACHE_DIR / "faces"
TMP_CACHE_DIR = CACHE_DIR / "tmp"
VIDEOS_CACHE_DIR = CACHE_DIR / "videos"
VIDEOS_RESULTS_DIR = VIDEOS_CACHE_DIR / "results"
# ===============================================================

SWAP_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
RESULT_LOCK = threading.Lock()
SSL_CONTEXT = ssl._create_unverified_context()
KNOWN_HOSTING_IP = "45.149.77.233"

LOGGER = logging.getLogger("reactor.swap")
if not LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _ensure_cache_dirs() -> None:
    SOURCES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TARGETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FACES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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

from scripts import reactor_swapper  # noqa: E402
reactor_swapper.FACES_CACHE_DIR = str(FACES_CACHE_DIR.resolve())
from scripts.reactor_swapper import analyze_faces, swap_face  # noqa: E402


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





def _build_direct_ip_url(path_or_url: str, direct_ip: str) -> tuple[str, str] | None:
    if not _is_url(path_or_url):
        return None

    parsed = urlparse(path_or_url)
    if not parsed.hostname:
        return None

    # فقط برای دامنه های مشخص شده از آی پی مستقیم استفاده شود
    target_domains = ['haani.ir', 'copypastekon.ir']
    if not any(domain in parsed.hostname for domain in target_domains):
        return None

    direct_url = parsed._replace(netloc=f"{direct_ip}:{parsed.port}" if parsed.port else direct_ip).geturl()
    return direct_url, parsed.hostname

def _resolve_tmp_cached_url_image(path_or_url: str) -> Path | None:
    if not _is_url(path_or_url):
        return None

    parsed_url = urlparse(path_or_url)
    raw_path = parsed_url.path
    if not raw_path:
        return None

    decoded_path = unquote(raw_path)
    candidate_paths = [Path(raw_path)]
    if decoded_path and decoded_path != raw_path:
        candidate_paths.append(Path(decoded_path))

    checked_rel_paths: list[Path] = []
    for candidate_path in candidate_paths:
        file_name = candidate_path.name
        if not file_name:
            continue

        rel_file_only = Path(file_name)
        if rel_file_only not in checked_rel_paths:
            checked_rel_paths.append(rel_file_only)

        parent_name = candidate_path.parent.name
        if parent_name:
            rel_with_parent = Path(parent_name) / file_name
            if rel_with_parent not in checked_rel_paths:
                checked_rel_paths.append(rel_with_parent)

    for rel_path in checked_rel_paths:
        candidate_path = TMP_CACHE_DIR / rel_path
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path

    return None

def _load_image(
    path_or_url: str,
    cache_subdir: Path,
    timing_prefix: str | None = None,
    timings_out: dict[str, float] | None = None,
) -> tuple[Image.Image, Path]:
    """
    Load the *current* image for a source/target reference.

    Cache policy:
    - Never trust URL/path identity as image identity.
    - Always resolve fresh bytes first, then cache by content hash.
    - This guarantees we only reuse cache when source/target content is truly unchanged.
    """
    def _mark(stage_name: str, stage_start: float) -> None:
        if timing_prefix is None or timings_out is None:
            return
        timings_out[f"{timing_prefix}_{stage_name}"] = (time.perf_counter() - stage_start) * 1000

    t_mkdir = time.perf_counter()
    cache_subdir.mkdir(parents=True, exist_ok=True)
    _mark("mkdir", t_mkdir)

    t_base64 = time.perf_counter()
    decoded_inline_image = _try_decode_base64_image(path_or_url)
    _mark("base64_decode", t_base64)
    if decoded_inline_image is not None:
        t_hash = time.perf_counter()
        content_hash = _image_sha256_hex(decoded_inline_image)
        _mark("hash", t_hash)
        cache_file = cache_subdir / f"{content_hash}.png"
        t_cache_hit_read = time.perf_counter()
        if cache_file.exists():
            cached_image = Image.open(cache_file).convert("RGB")
            _mark("cache_hit_read", t_cache_hit_read)
            return cached_image, cache_file
        t_cache_write = time.perf_counter()
        decoded_inline_image.save(cache_file, format="PNG")
        _mark("cache_write", t_cache_write)
        return decoded_inline_image, cache_file

    # IMPORTANT:
    # - query parsing already decodes URL parameters once.
    # - some CDNs include encoded characters inside path segments (e.g. %20).
    # If we unquote() a remote URL again, %20 turns into a literal space and
    # urllib raises "URL can't contain control characters".
    # So for HTTP(S), keep the URL as-is and do not unquote it again.
    if _is_url(path_or_url):
        t_url_lookup = time.perf_counter()
        url_key = _sha256_hex(path_or_url)
        url_map_dir = cache_subdir / "_url_index"
        url_map_dir.mkdir(parents=True, exist_ok=True)
        url_map_file = url_map_dir / f"{url_key}.txt"
        if url_map_file.exists():
            try:
                cached_hash = url_map_file.read_text(encoding="utf-8").strip()
                if cached_hash:
                    cache_file = cache_subdir / f"{cached_hash}.png"
                    if cache_file.exists():
                        cached_image = Image.open(cache_file).convert("RGB")
                        _mark("url_cache_hit_read", t_url_lookup)
                        return cached_image, cache_file
            except Exception:
                pass
        _mark("url_cache_lookup", t_url_lookup)

        t_tmp_lookup = time.perf_counter()
        tmp_cached_file = _resolve_tmp_cached_url_image(path_or_url)
        _mark("tmp_lookup", t_tmp_lookup)
        if tmp_cached_file is not None:
            t_tmp_open = time.perf_counter()
            image = Image.open(tmp_cached_file).convert("RGB")
            _mark("tmp_open", t_tmp_open)
            t_hash = time.perf_counter()
            content_hash = _image_sha256_hex(image)
            _mark("hash", t_hash)
            cache_file = cache_subdir / f"{content_hash}.png"
            t_cache_write = time.perf_counter()
            if not cache_file.exists():
                image.save(cache_file, format="PNG")
            _mark("cache_write_if_miss", t_cache_write)
            t_url_index_write = time.perf_counter()
            try:
                url_map_file.write_text(content_hash, encoding="utf-8")
            except Exception:
                pass
            _mark("url_cache_index_write", t_url_index_write)
            return image, cache_file

        t_download = time.perf_counter()
        parsed_url = urlparse(path_or_url)

        direct_ip_url_info = _build_direct_ip_url(path_or_url, KNOWN_HOSTING_IP)
        request_candidates: list[tuple[str, str | None]] = [(path_or_url, None)]
        if direct_ip_url_info is not None and direct_ip_url_info[0] != path_or_url:
            request_candidates.append((direct_ip_url_info[0], direct_ip_url_info[1]))

        def _download_with_urllib(request_url: str, host_header: str | None = None) -> bytes:
            req_headers = {"User-Agent": "ReActor-Standalone/1.0"}
            if host_header:
                req_headers["Host"] = host_header
            
            # Use system proxies for normal requests, bypass only for direct IP hits
            import urllib.request
            handlers = [urllib.request.HTTPSHandler(context=SSL_CONTEXT)]
            if host_header:
                # Force bypass proxies when hitting a direct IP to avoid routing issues
                handlers.append(urllib.request.ProxyHandler({}))
            
            opener = urllib.request.build_opener(*handlers)
            
            req = Request(request_url, headers=req_headers)
            with opener.open(req, timeout=60) as response:
                return response.read()

        data: bytes | None = None
        download_errors: list[str] = []

        # Prioritize direct IP to skip DNS (CURLOPT_RESOLVE style)
        request_candidates: list[tuple[str, str | None]] = []
        direct_ip_url_info = _build_direct_ip_url(path_or_url, KNOWN_HOSTING_IP)
        if direct_ip_url_info is not None:
            request_candidates.append((direct_ip_url_info[0], direct_ip_url_info[1]))
        
        # Original URL as fallback (though IP should work if server is up)
        if not request_candidates or request_candidates[0][0] != path_or_url:
            request_candidates.append((path_or_url, None))

        for request_url, request_host in request_candidates:
            try:
                data = _download_with_urllib(request_url, host_header=request_host)
                break
            except Exception as exc:
                download_errors.append(f"url={request_url} error={exc}")
                
                # If it's a direct IP URL that failed, it might be due to server SSL config
                # but with unverified context it should be fine. We still try the next candidate.
                continue

        if data is None:
            raise RuntimeError(
                "Failed to download image after trying all strategies. "
                f"original_url={path_or_url} attempts={'; '.join(download_errors)}"
            )
        _mark("download", t_download)

        t_decode = time.perf_counter()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        _mark("decode", t_decode)
        t_hash = time.perf_counter()
        content_hash = _image_sha256_hex(image)
        _mark("hash", t_hash)
        cache_file = cache_subdir / f"{content_hash}.png"
        t_cache_write = time.perf_counter()
        if not cache_file.exists():
            image.save(cache_file, format="PNG")
        _mark("cache_write_if_miss", t_cache_write)
        t_url_index_write = time.perf_counter()
        try:
            url_map_file.write_text(content_hash, encoding="utf-8")
        except Exception:
            pass
        _mark("url_cache_index_write", t_url_index_write)
        return image, cache_file

    t_unquote = time.perf_counter()
    val = unquote(path_or_url)
    _mark("unquote", t_unquote)
    img_path = Path(val)
    if not img_path.is_absolute():
        img_path = Path.cwd() / img_path
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    t_local_open = time.perf_counter()
    local_img = Image.open(img_path).convert("RGB")
    _mark("local_open", t_local_open)
    t_hash = time.perf_counter()
    content_hash = _image_sha256_hex(local_img)
    _mark("hash", t_hash)
    cache_file = cache_subdir / f"{content_hash}.png"
    t_cache_hit_read = time.perf_counter()
    if cache_file.exists():
        cached_image = Image.open(cache_file).convert("RGB")
        _mark("cache_hit_read", t_cache_hit_read)
        return cached_image, cache_file
    t_cache_write = time.perf_counter()
    local_img.save(cache_file, format="PNG")
    _mark("cache_write", t_cache_write)
    return local_img, cache_file


def _is_video(path_or_url: str) -> bool:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg"}
    parsed = urlparse(path_or_url)
    path_str = unquote(parsed.path).lower()
    return any(path_str.endswith(ext) for ext in video_extensions)


def _get_video_fps(video_path: Path) -> float:
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        if "/" in output:
            num, den = map(float, output.split("/"))
            return num / den
        return float(output)
    except Exception as exc:
        LOGGER.warning("Failed to get video FPS for %s: %s. Defaulting to 30.", video_path, exc)
        return 30.0


def _extract_frames(video_path: Path, output_dir: Path, max_frames: int | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Check if frames already exist
    existing_frames = sorted(output_dir.glob("frame_*.jpg"))
    if existing_frames:
        if max_frames is None or len(existing_frames) >= max_frames:
            return len(existing_frames)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vsync", "0",
        "-q:v", "2", # High quality JPEG
    ]
    if max_frames is not None:
        cmd.extend(["-vframes", str(max_frames)])
    
    cmd.append(str(output_dir / "frame_%05d.jpg"))
    
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        extracted = list(output_dir.glob("frame_*.jpg"))
        if not extracted:
             raise RuntimeError("ffmpeg finished but no frames were extracted")
        return len(extracted)
    except subprocess.CalledProcessError as exc:
        err_msg = exc.output.decode("utf-8", errors="ignore") if exc.output else str(exc)
        LOGGER.error("ffmpeg extraction failed: %s", err_msg)
        raise RuntimeError(f"Failed to extract frames: {err_msg}")


def _assemble_video(frames_dir: Path, output_path: Path, fps: float, original_video: Path | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    input_pattern = frames_dir / "frame_%05d.jpg"
    if not any(frames_dir.glob("frame_*.jpg")):
        raise RuntimeError(f"No frames found in {frames_dir} to assemble video")

    # Simple assembly without audio for now to keep it robust
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(input_pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if not output_path.exists():
            raise RuntimeError("ffmpeg finished but output video file was not created")
    except subprocess.CalledProcessError as exc:
        err_msg = exc.output.decode("utf-8", errors="ignore") if exc.output else str(exc)
        LOGGER.error("ffmpeg assembly failed: %s", err_msg)
        raise RuntimeError(f"Failed to assemble video: {err_msg}")


def _download_video(url: str, dest_path: Path) -> None:
    if dest_path.exists():
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use requests for simpler video downloading
    try:
        resp = requests.get(url, stream=True, timeout=600, verify=False)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as exc:
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download video from {url}: {exc}")


def _face_cache_path_for_target(target_cache_file: Path) -> Path:
    return target_cache_file.with_name(f"{target_cache_file.stem}_face.png")


def _face_position_cache_path_for_target(target_cache_file: Path) -> Path:
    return target_cache_file.with_name(f"{target_cache_file.stem}_face.txt")


def _read_face_position(path: Path) -> tuple[int, int, int, int] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        parts = [int(v.strip()) for v in raw.split(",")]
        if len(parts) != 4:
            return None
        left, top, right, bottom = parts
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom
    except Exception:
        return None


def _write_face_position(path: Path, crop_box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = crop_box
    path.write_text(f"{left},{top},{right},{bottom}", encoding="utf-8")


def _square_from_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1)
    side = max(1, int(round(side)))

    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        shift = right - width
        left -= shift
        right = width
    if bottom > height:
        shift = bottom - height
        top -= shift
        bottom = height

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)

    return left, top, right, bottom


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


def process_swap_request(path: str, query_params: dict[str, list[str]], request_id: str, output_format: str = "JPEG") -> tuple[int, dict[str, str], bytes]:
    """
    Core logic to handle a swap request (image or video).
    Returns (status_code, headers, body_bytes).
    """
    stage_times_ms: dict[str, float] = {}

    def mark(stage_name: str, stage_start: float) -> None:
        stage_times_ms[stage_name] = (time.perf_counter() - stage_start) * 1000

    if path not in {"/swap", "/swap_face_square"}:
        return 404, {}, b"Use /swap or /swap_face_square"

    only_face_square = path == "/swap_face_square"
    source_url = query_params.get("source_url", [None])[0]
    target_url = query_params.get("target_url", [None])[0]
    frames_param = query_params.get("frames", [None])[0]
    max_frames = int(frames_param) if frames_param and frames_param.isdigit() else None
    
    fps_param = query_params.get("fps", [None])[0]
    # As per user request: if frames is provided, use it as FPS too, unless explicit fps is given
    requested_fps = None
    if fps_param:
        try:
            requested_fps = float(fps_param)
        except ValueError:
            pass
    elif max_frames is not None:
        requested_fps = float(max_frames)

    # Allow requesting PNG format for internal sub-requests
    req_format = query_params.get("format", [output_format])[0].upper()
    if req_format not in {"JPEG", "PNG"}:
        req_format = "JPEG"

    if not source_url or not target_url:
        return 400, {}, b"source_url and target_url are required"

    try:
        t_stage = time.perf_counter()
        swap_options = _build_swap_options(query_params)
        mark("parse_options", t_stage)

        t_stage = time.perf_counter()
        source_img, _ = _load_image(
            source_url,
            SOURCES_CACHE_DIR,
            timing_prefix="source_load",
            timings_out=stage_times_ms,
        )
        mark("load_source", t_stage)

        if _is_video(target_url):
            t_video = time.perf_counter()
            target_id = _sha256_hex(target_url)
            video_cache_dir = VIDEOS_CACHE_DIR / target_id
            video_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Download or resolve local video
            if _is_url(target_url):
                video_path = video_cache_dir / "input_video.mp4"
                _download_video(target_url, video_path)
            else:
                video_path = Path(unquote(target_url))
                if not video_path.is_absolute():
                    video_path = Path.cwd() / video_path
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Cache key for the whole video result
            source_hash = _image_sha256_hex(source_img)
            cache_key = _sha256_hex(
                f"source={source_hash}|target_id={target_id}|opts={repr(sorted(swap_options.items()))}|model=default|frames={max_frames}"
            )
            result_video_path = VIDEOS_RESULTS_DIR / f"{cache_key}.mp4"
            
            with RESULT_LOCK:
                if result_video_path.exists():
                    body = result_video_path.read_bytes()
                    mark("video_cache_hit", t_video)
                    _log_timing(request_id, path, (time.perf_counter() - t_video), "video_cache_hit", stage_times_ms)
                    return 200, {"Content-Type": "video/mp4"}, body

            # Extraction
            frames_in_dir = video_cache_dir / "frames_in"
            _extract_frames(video_path, frames_in_dir, max_frames)
            
            # Processing
            frames_out_dir = video_cache_dir / f"frames_out_{cache_key}"
            # Cleanup old frames if they exist to avoid corruption from previous runs
            if frames_out_dir.exists():
                import shutil
                shutil.rmtree(frames_out_dir)
            frames_out_dir.mkdir(parents=True, exist_ok=True)
            
            input_frames = sorted(frames_in_dir.glob("frame_*.jpg"))
            if max_frames:
                input_frames = input_frames[:max_frames]
            
            # Determine the base URL for internal requests (Uvicorn port)
            internal_port = int(os.environ.get("REACTOR_PORT", "8008"))
            internal_base_url = f"http://127.0.0.1:{internal_port}/swap"

            def _process_frame(frame_path: Path):
                out_frame_path = frames_out_dir / frame_path.name
                if out_frame_path.exists():
                    return
                
                try:
                    if DISTRIBUTED_VIDEO_PROCESSING:
                        # Build query params for internal request
                        sub_query_params = {}
                        for key, vals in query_params.items():
                            if key not in ["source_url", "target_url", "frames", "format"]:
                                sub_query_params[key] = ",".join(vals)
                        
                        # Request JPEG to match our filename and preserve efficiency
                        sub_query_params["format"] = "JPEG"
                        sub_query_params["is_subrequest"] = "1"
                        
                        _, source_cache_file = _load_image(source_url, SOURCES_CACHE_DIR)
                        sub_query_params["source_url"] = str(source_cache_file.absolute())
                        sub_query_params["target_url"] = str(frame_path.absolute())
                        
                        resp = requests.get(internal_base_url, params=sub_query_params, timeout=60)
                        if resp.status_code == 200:
                            # Verify JPEG magic number (0xFF 0xD8)
                            if resp.content.startswith(b"\xff\xd8"):
                                with open(out_frame_path, "wb") as f:
                                    f.write(resp.content)
                            else:
                                LOGGER.error("Received non-JPEG content for frame %s (starts with: %s)", frame_path, resp.content[:8].hex())
                        else:
                            LOGGER.error("Distributed processing failed for frame %s: %s", frame_path, resp.text)
                    else:
                        frame_img = Image.open(frame_path).convert("RGB")
                        swapped_img, _, _ = swap_face(
                            source_img=source_img,
                            target_img=frame_img,
                            model=_pick_swap_model(),
                            **swap_options,
                        )
                        swapped_img.save(out_frame_path, format="JPEG", quality=95)
                except Exception as e:
                    LOGGER.error("Error processing frame %s: %s", frame_path, e)

            # Process frames in parallel
            list(SWAP_EXECUTOR.map(_process_frame, input_frames))
            
            # Assembly
            fps = _get_video_fps(video_path)
            _assemble_video(frames_out_dir, result_video_path, fps)
            
            body = result_video_path.read_bytes()
            mark("video_processed", t_video)
            _log_timing(request_id, path, (time.perf_counter() - t_video), "video_processed", stage_times_ms)
            return 200, {"Content-Type": "video/mp4"}, body

        else:
            t_stage = time.perf_counter()
            target_img, target_cache_file = _load_image(
                target_url,
                TARGETS_CACHE_DIR,
                timing_prefix="target_load",
                timings_out=stage_times_ms,
            )
            mark("load_target", t_stage)

            t_stage = time.perf_counter()
            source_hash = _image_sha256_hex(source_img)
            target_hash = _image_sha256_hex(target_img)
            target_face_cache_file = _face_cache_path_for_target(target_cache_file)
            target_face_position_file = _face_position_cache_path_for_target(target_cache_file)
            source_result_dir = RESULTS_CACHE_DIR / source_hash
            source_result_dir.mkdir(parents=True, exist_ok=True)
            cache_key = _sha256_hex(
                f"source={source_hash}|target={target_hash}|opts={repr(sorted(swap_options.items()))}|model=default|mode={'square' if only_face_square else 'full'}"
            )
            result_path = source_result_dir / f"{cache_key}.jpg"
            mark("prepare_cache_key", t_stage)

            def _run_swap() -> tuple[bytes, tuple[int, int, int, int] | None]:
                runtime_target_image = target_img
                used_face_cache = False
                cached_face_position = None
                run_swap_stage_times_ms: dict[str, float] = {}

                def run_swap_mark(stage_name: str, stage_start: float) -> None:
                    run_swap_stage_times_ms[stage_name] = (time.perf_counter() - stage_start) * 1000

                if only_face_square and target_face_cache_file.exists():
                    runtime_target_image = Image.open(target_face_cache_file).convert("RGB")
                    used_face_cache = True
                    cached_face_position = _read_face_position(target_face_position_file)
                    run_swap_stage_times_ms["face_square_cache_hit"] = 1.0
                elif only_face_square:
                    t_detect = time.perf_counter()
                    import cv2
                    import numpy as np

                    target_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
                    faces = analyze_faces(target_bgr)
                    run_swap_mark("detect_target_face", t_detect)
                    if len(faces) == 0:
                        raise RuntimeError("No target face found to crop")

                    largest_face = max(
                        faces,
                        key=lambda face: max(0.0, float(face.bbox[2] - face.bbox[0])) * max(0.0, float(face.bbox[3] - face.bbox[1])),
                    )
                    crop_box = _square_from_bbox(tuple(largest_face.bbox), target_img.width, target_img.height)
                    runtime_target_image = target_img.crop(crop_box)
                    cached_face_position = crop_box
                    used_face_cache = True
                    t_save_face_cache = time.perf_counter()
                    with RESULT_LOCK:
                        if not target_face_cache_file.exists():
                            runtime_target_image.save(target_face_cache_file, format="PNG")
                        if not target_face_position_file.exists():
                            _write_face_position(target_face_position_file, crop_box)
                    run_swap_mark("save_face_square_cache", t_save_face_cache)

                t_swap = time.perf_counter()
                swapped_img, bboxes, _ = swap_face(
                    source_img=source_img,
                    target_img=runtime_target_image,
                    model=_pick_swap_model(),
                    **swap_options,
                )
                run_swap_mark("swap", t_swap)

                crop_box = None
                output_image = swapped_img
                if only_face_square:
                    if used_face_cache:
                        crop_box = cached_face_position
                    else:
                        if not bboxes:
                            raise RuntimeError("No swapped target face found to crop")
                        crop_box = _square_from_bbox(tuple(bboxes[0]), swapped_img.width, swapped_img.height)
                        output_image = swapped_img.crop(crop_box)
                        with RESULT_LOCK:
                            if not target_face_cache_file.exists():
                                raw_target_face = target_img.crop(crop_box)
                                raw_target_face.save(target_face_cache_file, format="PNG")
                            if not target_face_position_file.exists():
                                _write_face_position(target_face_position_file, crop_box)

                output = io.BytesIO()
                t_encode = time.perf_counter()
                if req_format == "JPEG":
                    output_image.save(output, format="JPEG", quality=95)
                else:
                    output_image.save(output, format="PNG")
                body_inner = output.getvalue()
                run_swap_mark(f"encode_{req_format.lower()}", t_encode)

                t_write_result_cache = time.perf_counter()
                with RESULT_LOCK:
                    if not result_path.exists() and req_format == "JPEG":
                        result_path.write_bytes(body_inner)
                run_swap_mark("write_result_cache", t_write_result_cache)

                stage_times_ms.update(run_swap_stage_times_ms)
                return body_inner, crop_box

            t_cache_check = time.perf_counter()
            with RESULT_LOCK:
                if result_path.exists() and not only_face_square and req_format == "JPEG":
                    body = result_path.read_bytes()
                    mark("cache_lookup", t_cache_check)
                    _log_timing(request_id, path, (time.perf_counter() - t_stage), "cache_hit", stage_times_ms)
                    return 200, {"Content-Type": "image/jpeg"}, body
            mark("cache_lookup", t_cache_check)

            t_exec = time.perf_counter()
            body, crop_box = SWAP_EXECUTOR.submit(_run_swap).result()
            mark("executor_wait", t_exec)
            content_type = "image/png" if req_format == "PNG" else "image/jpeg"
            headers = {"Content-Type": content_type}
            if crop_box is not None:
                left, top, right, bottom = crop_box
                cookie_value = quote(f"x={left},y={top},w={right - left},h={bottom - top}")
                headers["Set-Cookie"] = f"swapped_face_pos={cookie_value}; Path=/; SameSite=Lax"
            
            _log_timing(request_id, path, (time.perf_counter() - t_stage), "processed", stage_times_ms)
            return 200, headers, body

    except Exception as exc:
        LOGGER.exception("[swap][error] req_id=%s path=%s exc=%s", request_id, path, exc)
        return 500, {}, str(exc).encode("utf-8", errors="ignore")


def _log_timing(request_id: str, path: str, elapsed_sec: float, outcome: str, stage_times_ms: dict[str, float]) -> None:
    elapsed_ms = elapsed_sec * 1000
    stage_parts = [f"{stage}={duration:.2f}ms" for stage, duration in sorted(stage_times_ms.items())]
    stages_str = " ".join(stage_parts)
    LOGGER.info("[swap] req_id=%s outcome=%s elapsed_ms=%.2f path=%s %s", request_id, outcome, elapsed_ms, path, stages_str)


class SwapHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        request_id = uuid.uuid4().hex[:8]
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        status, headers, body = process_swap_request(parsed.path, params, request_id)
        
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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
