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
import gc
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import requests
import cv2
import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image

# Force ORT memory efficiency if possible
import onnxruntime as ort

# ===== Standalone configuration (edit these paths directly) =====
REPO_ROOT = Path(__file__).resolve().parent
REPO_MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR = REPO_ROOT / "_models"
SWAP_MODEL_PATH = None
DEVICE_MODE = "gpu"
MAX_CONCURRENT_REQUESTS = 1 # Reducing to 1 for stability and VRAM check
DISTRIBUTED_VIDEO_PROCESSING = False

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
MODEL_LOCK = threading.Lock()
SSL_CONTEXT = ssl._create_unverified_context()
KNOWN_HOSTING_IP = "45.149.77.233"

# Memory cache for loaded images: {path_or_url: (Image, Path, content_hash, stat_or_None)}
_IMAGE_LOADER_CACHE: dict[str, tuple[Image.Image, Path, str, Any]] = {}
IMAGE_CACHE_MAX_ENTRIES = 16 # Aggressively small to save RAM

LOGGER = logging.getLogger("reactor.swap")
if not LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _ensure_cache_dirs() -> None:
    for d in [SOURCES_CACHE_DIR, TARGETS_CACHE_DIR, RESULTS_CACHE_DIR, FACES_CACHE_DIR, TMP_CACHE_DIR, VIDEOS_CACHE_DIR, VIDEOS_RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_sha256_hex(image: Image.Image) -> str:
    if hasattr(image, "reactor_hash"):
        return image.reactor_hash
    normalized = image.convert("RGB")
    h = hashlib.sha256(normalized.tobytes()).hexdigest()
    return h


def _install_comfy_stubs() -> None:
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
    folder_paths.get_full_path = lambda name, filename: os.path.join(models_dir, name, filename)

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    utils = types.ModuleType("comfy.utils")

    model_management.processing_interrupted = lambda: False
    def get_torch_device():
        requested = os.environ.get("REACTOR_DEVICE", DEVICE_MODE).lower()
        return "cuda" if requested == "gpu" else "cpu"
    model_management.get_torch_device = get_torch_device

    class ProgressBar:
        def __init__(self, total: int):
            self.total = max(int(total), 0)
            self.current = 0
        def update_absolute(self, value: int, total: int | None = None):
            if total is not None: self.total = max(int(total), 0)
            self.current = max(int(value), 0)
        def update(self, step: int = 1): self.current += int(step)

    utils.ProgressBar = ProgressBar
    utils.load_torch_file = lambda path, safe_load=True: torch.load(path, map_location="cpu")

    sys.modules["folder_paths"] = folder_paths
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.utils"] = utils


_ensure_cache_dirs()
_install_comfy_stubs()

from scripts import reactor_swapper
reactor_swapper.FACES_CACHE_DIR = str(FACES_CACHE_DIR.resolve())
from scripts.reactor_swapper import analyze_faces, swap_face

def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _try_decode_base64_image(value: str) -> Image.Image | None:
    candidate = value.strip()
    marker_idx = candidate.find("base64,")
    if marker_idx == -1: return None
    import base64
    payload = candidate[marker_idx + len("base64,") :]
    normalized_payload = "".join(payload.split())
    try:
        decoded = base64.b64decode(normalized_payload)
        img = Image.open(io.BytesIO(decoded)).convert("RGB")
        return img
    except Exception:
        return None

def _build_direct_ip_url(path_or_url: str, direct_ip: str) -> tuple[str, str] | None:
    if not _is_url(path_or_url): return None
    parsed = urlparse(path_or_url)
    if not parsed.hostname: return None
    target_domains = ['haani.ir', 'copypastekon.ir']
    if not any(domain in parsed.hostname for domain in target_domains): return None
    direct_url = parsed._replace(netloc=f"{direct_ip}:{parsed.port}" if parsed.port else direct_ip).geturl()
    return direct_url, parsed.hostname

def _resolve_tmp_cached_url_image(path_or_url: str) -> Path | None:
    if not _is_url(path_or_url): return None
    parsed_url = urlparse(path_or_url)
    raw_path = parsed_url.path
    if not raw_path: return None
    decoded_path = unquote(raw_path)
    candidate_paths = [Path(raw_path)]
    if decoded_path and decoded_path != raw_path: candidate_paths.append(Path(decoded_path))
    checked_rel_paths = []
    for cp in candidate_paths:
        if not cp.name: continue
        rel_file_only = Path(cp.name)
        if rel_file_only not in checked_rel_paths: checked_rel_paths.append(rel_file_only)
        if cp.parent.name:
            rel_with_parent = Path(cp.parent.name) / cp.name
            if rel_with_parent not in checked_rel_paths: checked_rel_paths.append(rel_with_parent)
    for rel_path in checked_rel_paths:
        candidate_path = TMP_CACHE_DIR / rel_path
        if candidate_path.exists() and candidate_path.is_file(): return candidate_path
    return None

def _load_image(path_or_url: str, cache_subdir: Path, timing_prefix: str | None = None, timings_out: dict[str, float] | None = None) -> tuple[Image.Image, Path]:
    def _mark(stage_name: str, stage_start: float) -> None:
        if timing_prefix is None or timings_out is None: return
        timings_out[f"{timing_prefix}_{stage_name}"] = (time.perf_counter() - stage_start) * 1000

    t_total = time.perf_counter()
    if path_or_url in _IMAGE_LOADER_CACHE:
        cached_img, cached_file, cached_hash, cached_stat = _IMAGE_LOADER_CACHE[path_or_url]
        if not _is_url(path_or_url):
            try:
                img_path = Path(unquote(path_or_url))
                if not img_path.is_absolute(): img_path = Path.cwd() / img_path
                current_stat = img_path.stat()
                if (cached_stat and current_stat.st_mtime == cached_stat.st_mtime and current_stat.st_size == cached_stat.st_size):
                    _mark("mem_cache_hit", t_total)
                    return cached_img, cached_file
            except Exception: pass
        else:
            _mark("mem_cache_hit", t_total)
            return cached_img, cached_file

    if len(_IMAGE_LOADER_CACHE) >= IMAGE_CACHE_MAX_ENTRIES:
        _IMAGE_LOADER_CACHE.clear()
        from scripts.reactor_swapper import clear_face_memory
        clear_face_memory()
        gc.collect()

    cache_subdir.mkdir(parents=True, exist_ok=True)
    decoded_inline_image = _try_decode_base64_image(path_or_url)
    if decoded_inline_image is not None:
        content_hash = _image_sha256_hex(decoded_inline_image)
        cache_file = cache_subdir / f"{content_hash}.png"
        if not cache_file.exists(): decoded_inline_image.save(cache_file, format="PNG")
        decoded_inline_image.reactor_hash = content_hash
        _IMAGE_LOADER_CACHE[path_or_url] = (decoded_inline_image, cache_file, content_hash, None)
        return decoded_inline_image, cache_file

    if _is_url(path_or_url):
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
                        cached_image.reactor_hash = cached_hash
                        _IMAGE_LOADER_CACHE[path_or_url] = (cached_image, cache_file, cached_hash, None)
                        return cached_image, cache_file
            except Exception: pass

        tmp_cached_file = _resolve_tmp_cached_url_image(path_or_url)
        if tmp_cached_file is not None:
            image = Image.open(tmp_cached_file).convert("RGB")
            content_hash = _image_sha256_hex(image)
            cache_file = cache_subdir / f"{content_hash}.png"
            if not cache_file.exists(): image.save(cache_file, format="PNG")
            try: url_map_file.write_text(content_hash, encoding="utf-8")
            except Exception: pass
            image.reactor_hash = content_hash
            _IMAGE_LOADER_CACHE[path_or_url] = (image, cache_file, content_hash, None)
            return image, cache_file

        # Download
        direct_ip_url_info = _build_direct_ip_url(path_or_url, KNOWN_HOSTING_IP)
        req_url = direct_ip_url_info[0] if direct_ip_url_info else path_or_url
        host_header = direct_ip_url_info[1] if direct_ip_url_info else None
        
        req_headers = {"User-Agent": "ReActor-Standalone/1.0"}
        if host_header: req_headers["Host"] = host_header
        
        resp = requests.get(req_url, headers=req_headers, timeout=60, verify=False)
        resp.raise_for_status()
        data = resp.content
        image = Image.open(io.BytesIO(data)).convert("RGB")
        content_hash = _image_sha256_hex(image)
        cache_file = cache_subdir / f"{content_hash}.png"
        if not cache_file.exists(): image.save(cache_file, format="PNG")
        try: url_map_file.write_text(content_hash, encoding="utf-8")
        except Exception: pass
        image.reactor_hash = content_hash
        _IMAGE_LOADER_CACHE[path_or_url] = (image, cache_file, content_hash, None)
        return image, cache_file

    img_path = Path(unquote(path_or_url))
    if not img_path.is_absolute(): img_path = Path.cwd() / img_path
    current_stat = img_path.stat()
    local_img = Image.open(img_path).convert("RGB")
    content_hash = _image_sha256_hex(local_img)
    cache_file = cache_subdir / f"{content_hash}.png"
    if not cache_file.exists(): local_img.save(cache_file, format="PNG")
    local_img.reactor_hash = content_hash
    _IMAGE_LOADER_CACHE[path_or_url] = (local_img, cache_file, content_hash, current_stat)
    return local_img, cache_file


def _load_video(path_or_url: str, cache_subdir: Path) -> Path:
    cache_subdir.mkdir(parents=True, exist_ok=True)
    
    if _is_url(path_or_url):
        url_key = _sha256_hex(path_or_url)
        url_map_dir = cache_subdir / "_url_index"
        url_map_dir.mkdir(parents=True, exist_ok=True)
        url_map_file = url_map_dir / f"{url_key}.txt"
        
        if url_map_file.exists():
            try:
                cached_filename = url_map_file.read_text(encoding="utf-8").strip()
                if cached_filename:
                    cache_file = cache_subdir / cached_filename
                    if cache_file.exists():
                        return cache_file
            except Exception: pass

        # Download
        direct_ip_url_info = _build_direct_ip_url(path_or_url, KNOWN_HOSTING_IP)
        req_url = direct_ip_url_info[0] if direct_ip_url_info else path_or_url
        host_header = direct_ip_url_info[1] if direct_ip_url_info else None
        
        req_headers = {"User-Agent": "ReActor-Standalone/1.0"}
        if host_header: req_headers["Host"] = host_header
        
        resp = requests.get(req_url, headers=req_headers, timeout=300, verify=False, stream=True)
        resp.raise_for_status()
        
        # Use content-disposition or URL path to get extension
        ext = ".mp4"
        parsed = urlparse(path_or_url)
        path_ext = Path(unquote(parsed.path)).suffix
        if path_ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg"}:
            ext = path_ext
            
        content_hash = hashlib.sha256()
        tmp_file = cache_subdir / f"tmp_{uuid.uuid4().hex}{ext}"
        with open(tmp_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                content_hash.update(chunk)
        
        final_hash = content_hash.hexdigest()
        cache_file = cache_subdir / f"{final_hash}{ext}"
        if cache_file.exists():
            tmp_file.unlink()
        else:
            tmp_file.rename(cache_file)
            
        try: url_map_file.write_text(cache_file.name, encoding="utf-8")
        except Exception: pass
        return cache_file

    v_path = Path(unquote(path_or_url))
    if not v_path.is_absolute(): v_path = Path.cwd() / v_path
    return v_path


def _is_video(path_or_url: str) -> bool:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg"}
    parsed = urlparse(path_or_url)
    return any(unquote(parsed.path).lower().endswith(ext) for ext in video_extensions)


def _face_cache_path_for_target(target_cache_file: Path) -> Path:
    return target_cache_file.with_name(f"{target_cache_file.stem}_face.png")

def _face_position_cache_path_for_target(target_cache_file: Path) -> Path:
    return target_cache_file.with_name(f"{target_cache_file.stem}_face.txt")

def _read_face_position(path: Path) -> tuple[int, int, int, int] | None:
    if not path.exists(): return None
    try:
        parts = [int(v.strip()) for v in path.read_text(encoding="utf-8").strip().split(",")]
        return tuple(parts) if len(parts) == 4 else None
    except Exception: return None

def _write_face_position(path: Path, crop_box: tuple[int, int, int, int]) -> None:
    path.write_text(f"{crop_box[0]},{crop_box[1]},{crop_box[2]},{crop_box[3]}", encoding="utf-8")

def _square_from_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(1, int(round(max(x2 - x1, y2 - y1))))
    left, top = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    right, bottom = left + side, top + side
    if left < 0: right -= left; left = 0
    if top < 0: bottom -= top; top = 0
    if right > width: shift = right - width; left -= shift; right = width
    if bottom > height: shift = bottom - height; top -= shift; bottom = height
    return max(0, left), max(0, top), min(width, right), min(height, bottom)

def _pick_swap_model() -> str:
    if SWAP_MODEL_PATH: return str(Path(SWAP_MODEL_PATH).resolve())
    for subdir in ("hyperswap", "reswapper", "insightface"):
        root = MODELS_DIR / subdir
        if not root.exists(): continue
        candidates = sorted(list(root.glob("*.onnx")) + list(root.glob("*.pth")))
        if candidates: return str(candidates[0])
    raise FileNotFoundError(f"No swap model found under {MODELS_DIR}")

def _build_swap_options(params: dict[str, list[str]]) -> dict:
    def _bool(name, default): return params.get(name, [str(default)])[0].strip().lower() in {"1", "true", "yes", "on"}
    def _int(name, default): return int(params.get(name, [str(default)])[0])
    def _list(name, default, t):
        raw = params.get(name, [None])[0]
        return [t(x.strip()) for x in raw.split(",") if x.strip()] if raw else default
    return {
        "source_faces_index": _list("source_faces_index", [0], int),
        "faces_index": _list("faces_index", [0], int),
        "gender_source": _int("gender_source", 0),
        "gender_target": _int("gender_target", 0),
        "faces_order": _list("faces_order", ["large-small", "large-small"], str),
        "face_boost_enabled": _bool("face_boost_enabled", False),
        "face_restore_model": params.get("face_restore_model", ["codeformer-v0.1.0.pth"])[0],
    }


def process_swap_request(path: str, query_params: dict[str, list[str]], request_id: str, output_format: str = "JPEG") -> tuple[int, dict[str, str], bytes]:
    stage_times_ms = {}
    def mark(name, start): stage_times_ms[name] = (time.perf_counter() - start) * 1000

    if path not in {"/swap", "/swap_face_square", "/build_source_cache"}:
        return 404, {}, b"Use /swap, /swap_face_square or /build_source_cache"

    if path == "/build_source_cache":
        source_url = query_params.get("source_url", [None])[0]
        if not source_url: return 400, {}, b"source_url is required"
        try:
            source_img, _ = _load_image(source_url, SOURCES_CACHE_DIR)
            with MODEL_LOCK:
                source_img_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
                faces = analyze_faces(source_img_cv)
                if faces:
                    from scripts.reactor_swapper import save_faces, get_image_md5hash, FACES_CACHE_DIR
                    if FACES_CACHE_DIR:
                        current_hash = get_image_md5hash(source_img)
                        face_cache_file = os.path.join(FACES_CACHE_DIR, f"{current_hash}.safetensors")
                        save_faces(faces, face_cache_file)
            return 200, {"Content-Type": "application/json"}, b'{"status": "success", "message": "Source face cached"}'
        except Exception as e: return 500, {}, str(e).encode("utf-8")

    source_url = query_params.get("source_url", query_params.get("source", [None]))[0]
    source_man_url = query_params.get("source_man", query_params.get("source_man_url", [None]))[0]
    target_url = query_params.get("target_url", query_params.get("target", [None]))[0]
    
    if not source_url or not target_url:
        return 400, {}, b"source_url and target_url are required"

    req_format = query_params.get("format", [output_format])[0].upper()
    if req_format not in {"JPEG", "PNG"}: req_format = "JPEG"

    try:
        t_total_start = time.perf_counter()
        swap_options = _build_swap_options(query_params)
        source_img, _ = _load_image(source_url, SOURCES_CACHE_DIR, "source_load", stage_times_ms)
        
        source_man_img = None
        if source_man_url:
            source_man_img, _ = _load_image(source_man_url, SOURCES_CACHE_DIR, "source_man_load", stage_times_ms)

        if _is_video(target_url):
            target_video_path = _load_video(target_url, VIDEOS_CACHE_DIR)
            source_hash = _image_sha256_hex(source_img)
            source_man_hash = _image_sha256_hex(source_man_img) if source_man_img else "none"
            video_name_hash = _sha256_hex(str(target_video_path))
            
            cache_key = _sha256_hex(f"src={source_hash}|man={source_man_hash}|v={video_name_hash}|opts={repr(sorted(swap_options.items()))}")
            result_video_path = VIDEOS_RESULTS_DIR / f"{cache_key}.mp4"
            
            with RESULT_LOCK:
                if result_video_path.exists():
                    _log_timing(request_id, path, time.perf_counter() - t_total_start, "video_cache_hit", stage_times_ms)
                    return 200, {"Content-Type": "video/mp4"}, result_video_path.read_bytes()

            def _run_video_swap_logic():
                cap = cv2.VideoCapture(str(target_video_path))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                tmp_output = result_video_path.with_suffix(".tmp.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(tmp_output), fourcc, fps, (width, height))
                
                model = _pick_swap_model()
                
                try:
                    for i in range(total_frames):
                        ret, frame = cap.read()
                        if not ret: break
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame_rgb)
                        
                        # Stage 1: Female / Unknown fallback
                        stage1_opts = swap_options.copy()
                        if source_man_img: stage1_opts["gender_target"] = 1
                        
                        with MODEL_LOCK:
                            swapped, _, _ = swap_face(source_img=source_img, target_img=pil_frame, model=model, **stage1_opts)
                        
                        # Stage 2: Male
                        if source_man_img:
                            stage2_opts = swap_options.copy()
                            stage2_opts["gender_target"] = 2
                            with MODEL_LOCK:
                                swapped, _, _ = swap_face(source_img=source_man_img, target_img=swapped, model=model, **stage2_opts)
                        
                        res_frame = cv2.cvtColor(np.array(swapped), cv2.COLOR_RGB2BGR)
                        out.write(res_frame)
                        
                        if i % 10 == 0:
                            from scripts.reactor_swapper import clear_face_memory
                            clear_face_memory()
                            gc.collect()
                            if torch.cuda.is_available(): torch.cuda.empty_cache()

                finally:
                    cap.release()
                    out.release()
                
                # Audio merge
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(tmp_output), "-i", str(target_video_path),
                        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        str(result_video_path)
                    ], check=True, capture_output=True)
                    if tmp_output.exists(): tmp_output.unlink()
                except Exception as e:
                    LOGGER.warning("FFmpeg failed or not found, returning video without audio: %s", e)
                    if tmp_output.exists(): tmp_output.rename(result_video_path)
                    
                return result_video_path.read_bytes()

            body = SWAP_EXECUTOR.submit(_run_video_swap_logic).result()
            _log_timing(request_id, path, time.perf_counter() - t_total_start, "video_processed", stage_times_ms)
            return 200, {"Content-Type": "video/mp4"}, body

        target_img, target_cache_file = _load_image(target_url, TARGETS_CACHE_DIR, "target_load", stage_times_ms)
        
        source_hash = _image_sha256_hex(source_img)
        source_man_hash = _image_sha256_hex(source_man_img) if source_man_img else "none"
        target_hash = _image_sha256_hex(target_img)
        
        target_face_cache_file = _face_cache_path_for_target(target_cache_file)
        target_face_position_file = _face_position_cache_path_for_target(target_cache_file)
        
        cache_key = _sha256_hex(f"src={source_hash}|man={source_man_hash}|tgt={target_hash}|opts={repr(sorted(swap_options.items()))}|mode={'sq' if only_face_square else 'fl'}")
        source_result_dir = RESULTS_CACHE_DIR / source_hash
        source_result_dir.mkdir(parents=True, exist_ok=True)
        result_path = source_result_dir / f"{cache_key}.jpg"

        with RESULT_LOCK:
            if result_path.exists() and req_format == "JPEG":
                body = result_path.read_bytes()
                crop_box = _read_face_position(target_face_position_file) if only_face_square else None
                headers = {"Content-Type": "image/jpeg"}
                if crop_box:
                    left, top, right, bottom = crop_box
                    cv = quote(f"x={left},y={top},w={right - left},h={bottom - top}")
                    headers["Set-Cookie"] = f"swapped_face_pos={cv}; Path=/; SameSite=Lax"
                _log_timing(request_id, path, time.perf_counter() - t_total_start, "cache_hit", stage_times_ms)
                return 200, headers, body

        def _run_swap_logic():
            rt_target_img = target_img
            cached_pos = None
            
            # STAGE 0: Optional Square Crop
            if only_face_square and target_face_cache_file.exists():
                rt_target_img = Image.open(target_face_cache_file).convert("RGB")
                cached_pos = _read_face_position(target_face_position_file)
            elif only_face_square:
                with MODEL_LOCK:
                    t_d = time.perf_counter()
                    target_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
                    faces = analyze_faces(target_bgr)
                    mark("detect", t_d)
                    if not faces: raise RuntimeError("No target face")
                    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    crop_box = _square_from_bbox(tuple(largest.bbox), target_img.width, target_img.height)
                    rt_target_img = target_img.crop(crop_box)
                    cached_pos = crop_box
                    with RESULT_LOCK:
                        if not target_face_cache_file.exists(): rt_target_img.save(target_face_cache_file, format="PNG")
                        if not target_face_position_file.exists(): _write_face_position(target_face_position_file, crop_box)
                    del target_bgr, faces

            # STAGE 1: Process primary source (Female / Unknown fallback if source_man is present)
            stage1_opts = swap_options.copy()
            if source_man_img:
                stage1_opts["gender_target"] = 1 # Female / Unknown
            
            with MODEL_LOCK:
                t_s1 = time.perf_counter()
                swapped, bboxes, _ = swap_face(source_img=source_img, target_img=rt_target_img, model=_pick_swap_model(), **stage1_opts)
                mark("swap_stage1", t_s1)

            # STAGE 2: Process source_man (Male targets only)
            final_swapped = swapped
            if source_man_img:
                stage2_opts = swap_options.copy()
                stage2_opts["gender_target"] = 2 # Male only
                with MODEL_LOCK:
                    t_s2 = time.perf_counter()
                    # We pass the result of stage 1 as the target for stage 2
                    final_swapped, bboxes_man, _ = swap_face(source_img=source_man_img, target_img=swapped, model=_pick_swap_model(), **stage2_opts)
                    mark("swap_stage2", t_s2)
                    # Correctly combine bboxes from both stages
                    if bboxes_man:
                        bboxes.extend(bboxes_man)

            out_img = final_swapped
            final_crop = None
            if only_face_square:
                if cached_pos: final_crop = cached_pos
                else:
                    # Logic to determine final crop box from processed faces if not cached
                    # Using bboxes from stage 1 or stage 2
                    active_bboxes = bboxes if bboxes else (bboxes_man if source_man_img and 'bboxes_man' in locals() else None)
                    if not active_bboxes: raise RuntimeError("No swapped face found for cropping")
                    final_crop = _square_from_bbox(tuple(active_bboxes[0]), final_swapped.width, final_swapped.height)
                    out_img = final_swapped.crop(final_crop)
                    with RESULT_LOCK:
                        if not target_face_cache_file.exists(): rt_target_img.save(target_face_cache_file, format="PNG")
                        if not target_face_position_file.exists(): _write_face_position(target_face_position_file, final_crop)

            out_io = io.BytesIO()
            out_img.save(out_io, format=req_format, quality=95 if req_format=="JPEG" else None)
            body = out_io.getvalue()
            if req_format == "JPEG":
                with RESULT_LOCK: result_path.write_bytes(body)
            
            if rt_target_img != target_img: del rt_target_img
            del swapped, final_swapped, out_img, out_io
            return body, final_crop

        body, crop_box = SWAP_EXECUTOR.submit(_run_swap_logic).result()
        headers = {"Content-Type": "image/png" if req_format=="PNG" else "image/jpeg"}
        if crop_box:
            l, t, r, b = crop_box
            cv = quote(f"x={l},y={t},w={r-l},h={b-t}")
            headers["Set-Cookie"] = f"swapped_face_pos={cv}; Path=/; SameSite=Lax"
        
        _log_timing(request_id, path, time.perf_counter() - t_total_start, "processed", stage_times_ms)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return 200, headers, body

    except Exception as exc:
        LOGGER.exception("[error] req_id=%s exc=%s", request_id, exc)
        return 500, {}, str(exc).encode("utf-8")


def _log_timing(request_id, path, elapsed_sec, outcome, stage_times):
    import psutil
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 / 1024
    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    stages = " ".join([f"{k}={v:.1f}ms" for k, v in sorted(stage_times.items())])
    LOGGER.info("[swap] id=%s outcome=%s ms=%.1f RAM=%.1fMB VRAM=%.1fMB path=%s %s", request_id, outcome, elapsed_sec*1000, ram_mb, vram_mb, path, stages)


class SwapHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        rid = uuid.uuid4().hex[:8]
        parsed = urlparse(self.path)
        status, headers, body = process_swap_request(parsed.path, parse_qs(parsed.query), rid)
        self.send_response(status)
        for k, v in headers.items(): self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def main():
    host, port = os.environ.get("REACTOR_HOST", "0.0.0.0"), int(os.environ.get("REACTOR_PORT", "8004"))
    server = ThreadingHTTPServer((host, port), SwapHandler)
    print(f"ReActor Standalone API: http://{host}:{port} | VRAM Caching Enabled")
    server.serve_forever()

if __name__ == "__main__": main()
