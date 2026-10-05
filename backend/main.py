import io
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import jobs
from jobs import (
    cancel_requested,
    create_job,
    finish_job,
    get_job,
    get_startup_progress,
    make_progress_callback,
    prune_jobs,
    request_cancel,
    update_job,
)
from model_service import (
    BreastCancerFuzzyEnsembleService,
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    CLASS_COLORS,
    CLASS_SEVERITY,
    FUZZY_DENSITIES,
    ROOT_DIR,
    DEFAULT_TEST_DIR
)

app = FastAPI(
    title="fuzzyBACH - Breast Cancer Histology Deep AI API",
    description="Backend API powered by 5 Deep Learning Backbones (Xception, VGG16, VGG19, InceptionV3, InceptionResNetV2) and Choquet Fuzzy Integral Decision Fusion for ICIAR BACH Histopathology Diagnosis.",
    version="2.0.0"
)

# Enable CORS for frontend development and local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
ASSETS_DIR = ROOT_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

if DEFAULT_TEST_DIR.exists():
    app.mount("/sample_photos", StaticFiles(directory=str(DEFAULT_TEST_DIR)), name="sample_photos")

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets_dist", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets_dist")

# Initialize 5-model Fuzzy Ensemble service
model_service = BreastCancerFuzzyEnsembleService()


class CsvExportRequest(BaseModel):
    items: List[dict]


@app.get("/")
def read_root():
    dist_index = FRONTEND_DIST / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index))
    return {
        "status": "online",
        "service": "fuzzyBACH Breast Cancer Histology Classification API",
        "models": ["Xception", "VGG16", "VGG19", "InceptionV3", "InceptionResNetV2", "Choquet Fuzzy Ensemble"],
        "ensemble_accuracy": "95.0% (4-class)",
        "docs": "/docs"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy" if model_service.is_loaded else "degraded",
        "model_loaded": model_service.is_loaded,
        "load_error": model_service.load_error,
        "models_count": 5,
        "ensemble_active": True,
        "classes": CLASS_NAMES,
        "startup": get_startup_progress(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/api/startup/progress")
def startup_progress():
    """Cold-start progress: which of the 5 backbones is loading right now."""
    prune_jobs()
    return get_startup_progress()


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    prune_jobs()
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    request_cancel(job_id)
    return {"job_id": job_id, "cancel_requested": True}


def _run_batch_job(job_id: str, image_items, mode: str):
    cb = make_progress_callback(job_id)
    try:
        update_job(job_id, status="running", stage="inference",
                   message=f"Running {mode} inference on {len(image_items)} image(s)…")
        result = model_service.predict_image_batch(
            image_items, mode=mode, progress_cb=cb,
            cancel_flag=lambda: cancel_requested(job_id))
        if cancel_requested(job_id):
            finish_job(job_id, cancelled=True)
        else:
            finish_job(job_id, result=result)
    except Exception as e:
        finish_job(job_id, error=str(e))


def _run_sample_job(job_id: str, limit: int, offset: int, mode: str):
    cb = make_progress_callback(job_id)
    try:
        update_job(job_id, status="running", stage="loading",
                   message=f"Loading sample dataset (limit={limit}, offset={offset})…")
        result = model_service.predict_sample_dataset(
            limit=limit, offset=offset, mode=mode, progress_cb=cb,
            cancel_flag=lambda: cancel_requested(job_id))
        if cancel_requested(job_id):
            finish_job(job_id, cancelled=True)
        else:
            finish_job(job_id, result=result)
    except Exception as e:
        finish_job(job_id, error=str(e))


@app.get("/api/model/info")
def get_model_info():
    return {
        "architecture": "Fuzzy Choquet Integral Decision Fusion over 5 Deep Learning Backbones",
        "backbones": [
            {"name": "Xception", "single_acc": 91.0, "weights": "xception_upto95frozen_weights.h5", "weight_dim": 2520},
            {"name": "InceptionResNetV2", "single_acc": 91.0, "weights": "incepresnet_temperature_normalized_at_3_0.5_dropout_weights.h5", "weight_dim": 464},
            {"name": "InceptionV3", "single_acc": 90.0, "weights": "inceptionv3_upto197frozen_weights.h5", "weight_dim": 2320},
            {"name": "VGG16", "single_acc": 86.0, "weights": "vgg16_upto15frozen_weights.h5", "weight_dim": 1152},
            {"name": "VGG19", "single_acc": 83.0, "weights": "vgg19_upto17frozen_weights.h5", "weight_dim": 896},
        ],
        "fuzzy_densities": FUZZY_DENSITIES,
        "input_resolution": "512 x 512 x 3 (RGB)",
        "preprocessing": "RGB bilinear resampling to 512x512 (dense heads were trained on Macenko-normalized data; no stain renormalization at inference)",
        "classes": [
            {
                "id": i,
                "name": name,
                "severity": CLASS_SEVERITY[name],
                "color": CLASS_COLORS[name],
                "description": CLASS_DESCRIPTIONS[name]
            }
            for i, name in enumerate(CLASS_NAMES)
        ],
        "benchmarks": {
            "two_class": [
                {"model": "VGG16", "val_acc": 100.0, "test_acc": 89.0},
                {"model": "VGG19", "val_acc": 99.8, "test_acc": 94.0},
                {"model": "Xception", "val_acc": 100.0, "test_acc": 95.0, "highlight": True},
                {"model": "Inception V3", "val_acc": 100.0, "test_acc": 94.0},
                {"model": "InceptionResNetV2", "val_acc": 99.7, "test_acc": 93.0},
                {"model": "Fuzzy Ensemble (Proposed)", "val_acc": "-", "test_acc": 96.0, "is_ensemble": True}
            ],
            "four_class": [
                {"model": "VGG16", "val_acc": 97.0, "test_acc": 86.0},
                {"model": "VGG19", "val_acc": 98.0, "test_acc": 83.0},
                {"model": "Xception", "val_acc": 99.0, "test_acc": 91.0, "highlight": True},
                {"model": "Inception V3", "val_acc": 99.0, "test_acc": 90.0},
                {"model": "InceptionResNetV2", "val_acc": 99.0, "test_acc": 91.0},
                {"model": "Fuzzy Choquet Ensemble (95% Acc)", "val_acc": "-", "test_acc": 95.0, "is_ensemble": True}
            ]
        },
        "citation": {
            "title": "Fuzzy ensemble of deep learning models using choquet fuzzy integral, coalition game and information theory for breast cancer histology classification",
            "authors": "Pratik Bhowal, Subhankar Sen, Juan D. Velasquez Silva, Ram Sarkar",
            "journal": "Expert Systems with Applications",
            "year": 2022,
            "publisher": "Elsevier",
            "doi_or_page": "116167"
        }
    }


@app.get("/api/samples")
def list_samples(limit: int = 20, offset: int = 0):
    if not DEFAULT_TEST_DIR.exists():
        return {"samples": [], "total": 0}

    image_paths = sorted(DEFAULT_TEST_DIR.glob("test*.tif"), key=lambda path: int(path.stem[4:]))
    total = len(image_paths)
    selected = image_paths[offset : offset + limit]

    samples_data = []
    for p in selected:
        samples_data.append({
            "filename": p.name,
            "id": int(p.stem[4:]),
            "size_bytes": p.stat().st_size,
            "preview_url": f"/sample_photos/{p.name}"
        })

    return {
        "samples": samples_data,
        "total": total,
        "offset": offset,
        "limit": limit
    }


@app.post("/api/predict/single")
async def predict_single(file: UploadFile = File(...), mode: str = Query("ensemble")):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
        result = model_service.predict_image_bytes(content, file.filename or "uploaded_image.tif", mode=mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), mode: str = Query("ensemble")):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        # Check if single zip file was provided
        if len(files) == 1 and files[0].filename.lower().endswith(".zip"):
            zip_bytes = await files[0].read()
            return model_service.predict_zip_archive(zip_bytes, mode=mode)

        # Process multiple files
        image_items = []
        for file in files:
            if file.filename.lower().endswith(".zip"):
                zip_content = await file.read()
                zip_res = model_service.predict_zip_archive(zip_content, mode=mode)
                return zip_res
            
            content = await file.read()
            if content:
                image_items.append((file.filename, content))

        if not image_items:
            raise HTTPException(status_code=400, detail="No valid image files found in upload.")

        return model_service.predict_image_batch(image_items, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.post("/api/predict/sample-dataset")
def predict_sample_dataset(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0), mode: str = Query("ensemble")):
    try:
        return model_service.predict_sample_dataset(limit=limit, offset=offset, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample dataset prediction error: {str(e)}")


@app.post("/api/predict/batch/async")
async def predict_batch_async(files: List[UploadFile] = File(...), mode: str = Query("ensemble")):
    """Non-blocking batch inference. Returns {job_id} immediately; poll GET /api/jobs/{job_id}.

    Sync POST /api/predict/batch is untouched for backward compatibility."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        # Single ZIP: expand in-request (fast) so the worker sees a concrete total.
        filenames = [(f.filename or "") for f in files]
        if len(files) == 1 and filenames[0].lower().endswith(".zip"):
            import zipfile as _zip
            zip_bytes = await files[0].read()
            image_items = []
            with _zip.ZipFile(io.BytesIO(zip_bytes)) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    from pathlib import Path as _P
                    if _P(info.filename).suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"} \
                            and not info.filename.startswith("__MACOSX"):
                        image_items.append((_P(info.filename).name, z.read(info.filename)))
            image_items.sort(key=lambda x: x[0])
        else:
            image_items = []
            for file in files:
                content = await file.read()
                if content:
                    image_items.append((file.filename or "upload", content))

        if not image_items:
            raise HTTPException(status_code=400, detail="No valid image files found in upload.")

        job_id = create_job("batch", len(image_items), mode)
        thread = threading.Thread(target=_run_batch_job, args=(job_id, image_items, mode), daemon=True)
        thread.start()
        return {"job_id": job_id, "total": len(image_items), "mode": mode}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Async batch submit error: {str(e)}")


@app.post("/api/predict/sample-dataset/async")
def predict_sample_dataset_async(limit: int = Query(10, ge=1, le=100), offset: int = Query(0, ge=0),
                                 mode: str = Query("ensemble")):
    """Non-blocking sample-dataset inference. Returns {job_id} immediately."""
    try:
        if not DEFAULT_TEST_DIR.exists():
            raise HTTPException(status_code=404, detail="Sample dataset not found on server.")
        image_paths = sorted(DEFAULT_TEST_DIR.glob("test*.tif"), key=lambda p: int(p.stem[4:]))
        total = len(image_paths[offset: offset + limit])
        if total == 0:
            raise HTTPException(status_code=404, detail="No sample images in requested range.")
        job_id = create_job("sample-dataset", total, mode)
        thread = threading.Thread(target=_run_sample_job, args=(job_id, limit, offset, mode), daemon=True)
        thread.start()
        return {"job_id": job_id, "total": total, "mode": mode, "limit": limit, "offset": offset}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Async sample submit error: {str(e)}")


@app.post("/api/export/csv")
def export_csv(payload: CsvExportRequest):
    try:
        if not payload.items:
            raise HTTPException(status_code=400, detail="No items to export.")
        csv_content = model_service.generate_csv(payload.items)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=fuzzyBACH_Ensemble_Predictions_{int(time.time())}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV Export error: {str(e)}")
