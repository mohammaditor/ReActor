import os
import shutil
import gc
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from reactor_core.analyzer import ReActorFaceAnalysis
from reactor_core.face_objects import Face
from reactor_core.inswap import INSwapper
from reactor_core.hyperswap import HyperSwapper
import torch

import folder_paths
import comfy.model_management as model_management
from r_modules.shared import state

from scripts.reactor_logger import logger
from reactor_utils import (
    move_path,
    get_image_md5hash,
    progress_bar,
    progress_bar_reset,
    save_faces,
    load_faces
)
from scripts.r_faceboost import swapper, restorer

import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')

# PROVIDERS
try:
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    elif torch.backends.mps.is_available():
        providers = ["CoreMLExecutionProvider"]
    elif hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
        providers = ["ROCMExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
except Exception as e:
    logger.debug(f"ExecutionProviderError: {e}.\nEP is set to CPU.")
    providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")
reswapper_path = os.path.join(models_path, "reswapper")
hyperswap_path = os.path.join(models_path, "hyperswap")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
    shutil.rmtree(insightface_path_old)
    shutil.rmtree(models_path_old)


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

FACES_CACHE_DIR = None

def unload_model(model):
    if model is not None:
        del model
    return None

def unload_all_models():
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    FS_MODEL = unload_model(FS_MODEL)
    ANALYSIS_MODELS["320"] = unload_model(ANALYSIS_MODELS["320"])
    ANALYSIS_MODELS["640"] = unload_model(ANALYSIS_MODELS["640"])

def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

def getAnalysisModel(det_size = (640, 640)):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = ReActorFaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    if FS_MODEL is None or CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = unload_model(FS_MODEL)

        model_filename = os.path.basename(model_path)
        if "hyperswap" in model_filename.lower(): # Если это Hyperswap
            model_path = os.path.join(folder_paths.models_dir, "hyperswap", model_filename)
            FS_MODEL = HyperSwapper(model_path, providers=providers)
        else: # Если это INSwapper / Reswapper
            if "reswapper" in model_filename.lower():
                model_path = os.path.join(folder_paths.models_dir, "reswapper", model_filename)
            FS_MODEL = INSwapper(model_path, providers=providers)

    return FS_MODEL

def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse = True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse = True)
    if order == "small-large":
        return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    # by default "large-small":
    return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)

def get_face_gender(
    face,
    face_index,
    gender_condition,
    operated: str,
    order: str,
):
    # 1. Сортируем ВСЕ найденные лица (без фильтрации!)
    faces_sorted = sort_by_order(face, order)

    # 2. Проверяем, существует ли вообще лицо с таким визуальным индексом
    if face_index >= len(faces_sorted):
        logger.info("Requested face index (%s) is out of bounds (max available index is %s)", face_index, len(faces_sorted) - 1)
        return None, 0, None

    # 3. Берем конкретное лицо по его позиции на фото (например, второе справа)
    face_selected = faces_sorted[face_index]

    # Если фильтр по полу отключен (no) - сразу отдаем лицо в работу
    if gender_condition == 0:
        return face_selected, 0, face_index

    # 4. Проверяем пол выбранного лица
    # face.gender: 0 = female, 1 = male
    # gender_condition: 1 = female, 2 = male
    expected_gender = 0 if gender_condition == 1 else 1
    actual_gender = getattr(face_selected, 'gender', -1)
    
    sel_gender_str = "Male" if actual_gender == 1 else "Female" if actual_gender == 0 else "Unknown"
    logger.info("%s Face %s: Detected Gender -%s-", operated, face_index, sel_gender_str)

    # Если пол не совпадает с тем, что заказал юзер
    if actual_gender != expected_gender:
        logger.info(f"{operated} Face {face_index}: WRONG gender ({sel_gender_str})")
        return face_selected, 1, face_index  # 1 означает флаг wrong_gender = True (цикл его пропустит)

    # Если всё идеально
    return face_selected, 0, face_index

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = getAnalysisModel(det_size)

    faces = []
    try:
        faces = face_analyser.get(img_data)
    except Exception as e:
        logger.error("No faces found")

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return analyze_faces(img_data, det_size_half)

    return faces

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, order="large-small"):

    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_source,"Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_target,"Target", order)
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)

    try:
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 0, face_index
    except IndexError:
        return None, 0, None


def clear_face_memory():
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH
    SOURCE_FACES = None
    SOURCE_IMAGE_HASH = None
    TARGET_FACES = None
    TARGET_IMAGE_HASH = None
    TARGET_FACES_LIST = []
    TARGET_IMAGE_LIST_HASH = []
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def swap_face(
    source_img: Union[Image.Image, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: Union[List[int], None] = None,
    faces_index: Union[List[int], None] = None,
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: Union[List[str], None] = None,
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    result_image = target_img
    bbox = []
    swapped_indexes = []

    source_faces_index = source_faces_index or [0]
    faces_index = faces_index or [0]
    faces_order = faces_order or ["large-small", "large-small"]

    if model is not None:
        if isinstance(source_img, str): 
            import base64, io
            base64_data = source_img.split('base64,')[-1]
            img_bytes = base64.b64decode(base64_data)
            source_img = Image.open(io.BytesIO(img_bytes))
            
        if source_img is not None:
            global SOURCE_IMAGE_HASH, SOURCE_FACES
            current_hash = get_image_md5hash(source_img)
            source_img_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
            if SOURCE_IMAGE_HASH != current_hash:
                face_cache_file = os.path.join(FACES_CACHE_DIR, f"{current_hash}.safetensors") if FACES_CACHE_DIR else None
                loaded = load_faces(face_cache_file) if face_cache_file else None
                if loaded is not None:
                    SOURCE_FACES = loaded
                    logger.status(f"Using Disk Cached Source Faces ({len(SOURCE_FACES)} found)...")
                    SOURCE_IMAGE_HASH = current_hash
                
                if SOURCE_IMAGE_HASH != current_hash:
                    logger.status("Analyzing Source Image...")
                    SOURCE_FACES = analyze_faces(source_img_cv)
                    SOURCE_IMAGE_HASH = current_hash
                    if face_cache_file:
                        save_faces(SOURCE_FACES, face_cache_file)
            else:
                logger.status("Using Memory Cached Source Faces...")
            source_faces = SOURCE_FACES
            del source_img_cv
        elif face_model is not None:
            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_faces = [face_model]
        else:
            logger.error("Cannot detect any Source")
            return result_image, bbox, swapped_indexes

        if source_faces is not None:
            global TARGET_IMAGE_HASH, TARGET_FACES
            target_hash = get_image_md5hash(target_img)
            target_img_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
            if TARGET_IMAGE_HASH != target_hash:
                target_face_cache_file = os.path.join(FACES_CACHE_DIR, f"{target_hash}.safetensors") if FACES_CACHE_DIR else None
                loaded = load_faces(target_face_cache_file) if target_face_cache_file else None
                if loaded is not None:
                    TARGET_FACES = loaded
                    logger.status(f"Using Disk Cached Target Faces ({len(TARGET_FACES)} found)...")
                    TARGET_IMAGE_HASH = target_hash
                
                if TARGET_IMAGE_HASH != target_hash:
                    logger.status("Analyzing Target Image...")
                    TARGET_FACES = analyze_faces(target_img_cv)
                    TARGET_IMAGE_HASH = target_hash
                    if target_face_cache_file:
                        save_faces(TARGET_FACES, target_face_cache_file)
            else:
                logger.status("Using Memory Cached Target Faces...")
            target_faces = TARGET_FACES

            if not target_faces:
                logger.status("Cannot detect any Target")
                del target_img_cv
                return result_image, bbox, swapped_indexes

            valid_source_faces = []
            # Simplified face picking
            for idx in source_faces_index:
                sf, wrong, _ = get_face_single(None, source_faces, face_index=idx, gender_source=gender_source, order=faces_order[1])
                if sf and wrong == 0: valid_source_faces.append(sf)

            if not valid_source_faces:
                logger.status("No valid source face(s)")
            else:
                result = target_img_cv
                face_swapper = getFaceSwapModel(model)
                source_face_idx = 0
                for face_num in faces_index:
                    target_face, wrong_gender, target_face_index = get_face_single(target_img_cv, target_faces, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                    if target_face and wrong_gender == 0:
                        logger.status(f"Swapping...")
                        source_face_to_use = valid_source_faces[source_face_idx % len(valid_source_faces)]
                        if face_boost_enabled and "hyperswap" not in model:
                            bgr_fake, M = face_swapper.get(result, target_face, source_face_to_use, paste_back=False)
                            bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                            M *= scale
                            result = swapper.in_swap(result, bgr_fake, M)
                        else:
                            result = face_swapper.get(result, target_face, source_face_to_use)
                        bbox.append(tuple(map(float, target_face.bbox)))
                        swapped_indexes.append(target_face_index)
                        if len(valid_source_faces) > 1: source_face_idx += 1
                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            del target_img_cv
    return result_image, bbox, swapped_indexes

def swap_face_many(
    source_img: Union[Image.Image, None],
    target_imgs: List[Image.Image],
    model: Union[str, None] = None,
    source_faces_index: Union[List[int], None] = None,
    faces_index: Union[List[int], None] = None,
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: Union[List[str], None] = None,
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    # This function is not optimized for memory here, but run.py disables video processing anyway
    return [target_img for target_img in target_imgs], [], []
