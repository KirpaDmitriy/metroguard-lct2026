from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from demo_app.algorithms import ALGORITHMS
from demo_app.config import ROOT, Settings, load_settings
from demo_app.job_store import cleanup_jobs, read_json, reconcile_jobs, write_json
from demo_app.storage import UploadRejected, save_upload


STATIC = Path(__file__).with_name("static")
PRESETS = Path(__file__).with_name("presets")
DEMO = ROOT / "demo"
METRICS = ROOT / "experiments/dashboard.html"


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or load_settings()
    config.work_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        reconcile_jobs(config.work_dir)
        cleanup_jobs(config.work_dir, config.job_retention_seconds)
        cleanup_task = asyncio.create_task(cleanup_loop(config))
        try:
            yield
        finally:
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
            tasks = list(application.state.tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="MetroGuard", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.settings = config
    app.state.tasks = {}
    app.state.capacity = asyncio.Semaphore(2)
    app.state.pending_jobs = 0
    app.state.pending_lock = asyncio.Lock()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    if DEMO.is_dir():
        app.mount("/demo", StaticFiles(directory=DEMO), name="demo")

    @app.middleware("http")
    async def reject_upload_when_busy(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/api/jobs":
            async with app.state.pending_lock:
                if app.state.pending_jobs >= config.max_pending_jobs:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "The analysis queue is full"},
                        headers={"Retry-After": "30"},
                    )
        return await call_next(request)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/metrics")
    async def metrics():
        if not METRICS.is_file():
            raise HTTPException(404, "Metrics dashboard is not available")
        return FileResponse(METRICS)

    @app.get("/api/config")
    async def api_config():
        return {
            "algorithms": ["best", *ALGORITHMS],
            "best_algorithm": config.best_algorithm,
            "max_upload_bytes": config.max_upload_bytes,
            "max_pending_jobs": config.max_pending_jobs,
        }

    @app.get("/api/presets")
    async def api_presets():
        return read_json(PRESETS / "manifest.json")["presets"]

    @app.get("/api/presets/{preset_id}")
    async def api_preset(preset_id: str, algorithm: str = "best"):
        selected = config.best_algorithm if algorithm == "best" else algorithm
        if selected not in ALGORITHMS:
            raise HTTPException(422, "Unknown algorithm")
        manifest = read_json(PRESETS / "manifest.json")["presets"]
        item = next((entry for entry in manifest if entry["id"] == preset_id), None)
        if item is None:
            raise HTTPException(404, "Preset not found")
        path = PRESETS / preset_id / f"{selected}.json"
        if not path.is_file():
            raise HTTPException(404, "Preset result not found")
        return {"preset": item, "result": read_json(path)}

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        bag: UploadFile = File(...),
        algorithm: str = Form("best"),
    ):
        selected = config.best_algorithm if algorithm == "best" else algorithm
        if selected not in ALGORITHMS:
            raise HTTPException(422, "Unknown algorithm")
        if not await reserve_job_slot(app):
            raise HTTPException(
                429,
                "The analysis queue is full",
                headers={"Retry-After": "30"},
            )
        job_id = uuid.uuid4().hex
        job_dir = config.work_dir / job_id
        input_path = job_dir / "input.db3"
        try:
            size = await save_upload(bag, input_path, config.max_upload_bytes)
        except UploadRejected as error:
            await release_job_slot(app)
            raise HTTPException(400, str(error)) from error
        except BaseException:
            await release_job_slot(app)
            raise
        status_path = job_dir / "status.json"
        write_json(status_path, {
            "status": "queued",
            "job_id": job_id,
            "algorithm": selected,
            "requested_algorithm": algorithm,
            "upload_bytes": size,
        })
        task = asyncio.create_task(run_reserved_job(app, job_id, selected))
        app.state.tasks[job_id] = task
        task.add_done_callback(lambda _task: app.state.tasks.pop(job_id, None))
        return {"job_id": job_id, "status": "queued", "algorithm": selected}

    @app.get("/api/jobs/{job_id}")
    async def job_status(job_id: str):
        job_dir = resolve_job_dir(config.work_dir, job_id)
        result = job_dir / "result.json"
        status = read_job_json(job_dir / "status.json")
        progress = job_dir / "progress.json"
        if status.get("status") == "running" and progress.is_file():
            status.update(read_job_json(progress))
            status.update(job_id=job_id, algorithm=status.get("algorithm", ""))
        if result.is_file():
            status["result_url"] = f"/api/jobs/{job_id}/result"
        return status

    @app.get("/api/jobs/{job_id}/result")
    async def job_result(job_id: str):
        path = resolve_job_dir(config.work_dir, job_id) / "result.json"
        if not path.is_file():
            raise HTTPException(409, "Job has not completed")
        return read_job_json(path)

    return app


def resolve_job_dir(work_dir: Path, job_id: str) -> Path:
    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
        raise HTTPException(404, "Job not found")
    path = work_dir / job_id
    if not path.is_dir():
        raise HTTPException(404, "Job not found")
    return path


def read_job_json(path: Path) -> dict:
    try:
        return read_json(path)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(404, "Job not found") from error


async def reserve_job_slot(app: FastAPI) -> bool:
    async with app.state.pending_lock:
        if app.state.pending_jobs >= app.state.settings.max_pending_jobs:
            return False
        app.state.pending_jobs += 1
        return True


async def release_job_slot(app: FastAPI) -> None:
    async with app.state.pending_lock:
        app.state.pending_jobs = max(0, app.state.pending_jobs - 1)


async def run_reserved_job(app: FastAPI, job_id: str, algorithm: str) -> None:
    try:
        await run_job(app, job_id, algorithm)
    finally:
        await release_job_slot(app)


async def cleanup_loop(settings: Settings) -> None:
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        cleanup_jobs(settings.work_dir, settings.job_retention_seconds)


async def run_job(app: FastAPI, job_id: str, algorithm: str) -> None:
    settings: Settings = app.state.settings
    job_dir = settings.work_dir / job_id
    status_path = job_dir / "status.json"
    progress_path = job_dir / "progress.json"
    output_path = job_dir / "result.json"
    command = [
        sys.executable,
        "-m",
        "demo_app.worker",
        "--input", str(job_dir / "input.db3"),
        "--algorithm", algorithm,
        "--output", str(output_path),
        "--progress", str(progress_path),
    ]
    async with app.state.capacity:
        write_json(status_path, {
            "status": "running", "job_id": job_id, "algorithm": algorithm
        })
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(), settings.job_timeout_seconds
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            write_json(status_path, {
                "status": "failed", "job_id": job_id,
                "algorithm": algorithm, "error": "Analysis timed out",
            })
            return
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            write_json(status_path, {
                "status": "failed", "job_id": job_id,
                "algorithm": algorithm, "error": "Server stopped during analysis",
            })
            raise
        except OSError as error:
            write_json(status_path, {
                "status": "failed", "job_id": job_id,
                "algorithm": algorithm, "error": str(error),
            })
            return
        if process.returncode:
            message = stderr.decode("utf-8", errors="replace")[-2000:]
            write_json(status_path, {
                "status": "failed", "job_id": job_id,
                "algorithm": algorithm,
                "error": message or f"Worker exited with code {process.returncode}",
            })
            return
        write_json(status_path, {
            "status": "complete", "job_id": job_id, "algorithm": algorithm,
            "result_url": f"/api/jobs/{job_id}/result",
        })


app = create_app()
