"""fuzzyBACH job + startup progress store.

Additive, non-breaking helper shared by model_service.py and main.py.
All state lives in-memory; old sync endpoints never touch it unless
a progress callback is explicitly supplied.
"""

import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

_lock = threading.Lock()

# --- Cold-start progress (updated by model_service._init_all_models) ---
STARTUP_STAGES = [
    "Xception",
    "VGG16",
    "VGG19",
    "InceptionV3",
    "InceptionResNetV2",
    "Warmup",
]

STARTUP_PROGRESS: Dict[str, Any] = {
    "stage": "queued",
    "stage_index": 0,
    "stage_total": len(STARTUP_STAGES),
    "current_model": None,
    "message": "Waiting for backend to start loading models…",
    "done": False,
    "error": None,
    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    """Backend-terminal log with immediate flush (visible in run_app.py console)."""
    print(message, flush=True)


def set_startup_progress(stage_index: int, current_model: Optional[str], message: str,
                         done: bool = False, error: Optional[str] = None) -> None:
    with _lock:
        STARTUP_PROGRESS.update({
            "stage": "done" if done and not error else ("error" if error else "loading"),
            "stage_index": stage_index,
            "stage_total": len(STARTUP_STAGES),
            "current_model": current_model,
            "message": message,
            "done": done,
            "error": error,
            "updated_at": _stamp(),
        })
    log(f"[Startup {stage_index}/{len(STARTUP_STAGES)}] {message}")


def get_startup_progress() -> Dict[str, Any]:
    with _lock:
        return dict(STARTUP_PROGRESS)


# --- Inference jobs (polled by the frontend) ---
JOBS: Dict[str, Dict[str, Any]] = {}
JOB_TTL_SECONDS = 30 * 60  # keep finished jobs for inspection/export


def create_job(kind: str, total: int, mode: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock:
        JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",  # queued | running | succeeded | failed | cancelled
            "stage": "queued",
            "current": 0,
            "total": total,
            "percent": 0.0,
            "current_image": None,
            "current_model": None,
            "message": f"Queued {total} image(s) for {mode} inference…",
            "mode": mode,
            "cancel_requested": False,
            "result": None,
            "error": None,
            "created_at": _stamp(),
            "updated_at": _stamp(),
            "_created_ts": now,
        }
    log(f"[Job {job_id}] queued kind={kind} total={total} mode={mode}")
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = JOBS.get(job_id)
        if job is None:
            return None
        # Hide internal bookkeeping key from API responses.
        return {k: v for k, v in job.items() if not k.startswith("_")}


def request_cancel(job_id: str) -> bool:
    with _lock:
        job = JOBS.get(job_id)
        if job is None:
            return False
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return True
        job["cancel_requested"] = True
        job["updated_at"] = _stamp()
        return True


def cancel_requested(job_id: str) -> bool:
    with _lock:
        job = JOBS.get(job_id)
        return bool(job and job.get("cancel_requested"))


def update_job(job_id: str, **fields: Any) -> None:
    with _lock:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = _stamp()
        total = job.get("total") or 0
        current = job.get("current") or 0
        job["percent"] = round((current / total) * 100, 1) if total else 0.0


def finish_job(job_id: str, result: Optional[Dict[str, Any]] = None,
               error: Optional[str] = None, cancelled: bool = False) -> None:
    with _lock:
        job = JOBS.get(job_id)
        if job is None:
            return
        if cancelled or job.get("cancel_requested"):
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["message"] = "Cancelled by user."
        elif error:
            job["status"] = "failed"
            job["stage"] = "failed"
            job["error"] = error
            job["message"] = f"Failed: {error}"
        else:
            job["status"] = "succeeded"
            job["stage"] = "done"
            job["result"] = result
            job["message"] = f"Done — {job.get('current', 0)}/{job.get('total', 0)} images."
            if job.get("total"):
                job["percent"] = 100.0
        job["updated_at"] = _stamp()
    # Log outside the lock to avoid re-entrancy.
    status = (JOBS.get(job_id) or {}).get("status", "?")
    log(f"[Job {job_id}] {status}: {((JOBS.get(job_id) or {}).get('message'))}")


def make_progress_callback(job_id: str) -> Callable[..., None]:
    """Callback passed into model_service batch loops.

    Signature: cb(done_count, total, filename, model_name, stage)
    Safe to call from the worker thread; never raises.
    """

    def _cb(done_count: int, total: int, filename: Optional[str] = None,
            model_name: Optional[str] = None, stage: Optional[str] = None) -> None:
        try:
            if cancel_requested(job_id):
                return
            msg = f"Image {done_count}/{total}"
            if filename:
                msg += f" — {filename}"
            if model_name:
                msg += f" [{model_name}]"
            update_job(
                job_id,
                status="running",
                stage=stage or "inference",
                current=done_count,
                total=total,
                current_image=filename,
                current_model=model_name,
                message=msg,
            )
            # Light terminal heartbeat: first image, every 10th, and last.
            if done_count in (1, total) or done_count % 10 == 0:
                log(f"[Job {job_id}] {msg}")
        except Exception:
            pass

    return _cb


def prune_jobs() -> None:
    now = time.time()
    with _lock:
        stale = [jid for jid, j in JOBS.items()
                 if j.get("status") in ("succeeded", "failed", "cancelled")
                 and (now - j.get("_created_ts", now)) > JOB_TTL_SECONDS]
        for jid in stale:
            JOBS.pop(jid, None)
