"""Reels Studio web server."""
import logging
import os
import queue
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import core
import picker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

DATA = Path(os.environ.get("DATA_DIR", "/data/jobs"))
JOB_TTL = float(os.environ.get("JOB_TTL_HOURS", "24")) * 3600
STATIC = Path(__file__).with_name("static")

# Stage -> (start %, end %) of the overall bar
STAGES = {"download": (0, 15), "transcribe": (15, 55), "pick": (55, 60), "render": (60, 100)}

app = FastAPI(title="Reels Studio")
jobs: dict = {}
GPU: dict = {}
jobs_lock = threading.Lock()
work = queue.Queue()


class Job:
    def __init__(self, url, upload_name, layouts, captions, use_llm):
        self.id = uuid.uuid4().hex
        self.dir = DATA / self.id
        self.dir.mkdir(parents=True)
        self.url = url
        self.upload_name = upload_name
        self.source = None
        self.layouts = layouts
        self.captions = captions
        self.use_llm = use_llm
        self.status = "queued"
        self.stage = "download" if url else "transcribe"
        self.percent = 0.0
        self.message = "Waiting to start"
        self.updates = []
        self.clips = []
        self.error = None
        self.cancelled = False
        self.created = time.time()
        self.finished = None

    def note(self, msg):
        self.message = msg
        self.updates.append({"t": time.time(), "msg": msg})
        self.updates = self.updates[-30:]
        log.info("[%s] %s", self.id[:8], msg)

    def progress(self, stage, frac):
        if self.cancelled:
            raise core.Cancelled()
        a, b = STAGES[stage]
        self.stage = stage
        self.percent = max(self.percent, a + (b - a) * frac)

    def public(self):
        return {
            "id": self.id, "status": self.status, "stage": self.stage,
            "percent": round(self.percent, 1), "message": self.message,
            "updates": self.updates[-8:], "error": self.error,
            "layouts": self.layouts, "captions": self.captions, "use_llm": self.use_llm,
            "clips": self.clips,
        }


def run_job(job):
    job.status = "running"
    try:
        if job.url:
            job.note("Downloading video")
            last = [0]

            def dl_progress(f):
                job.progress("download", f)
                pct = int(f * 100)
                if pct >= last[0] + 25:
                    last[0] = pct
                    job.note(f"Downloading video ({pct}%)")
            job.source, title = core.download(job.url, job.dir, dl_progress)
        else:
            title = Path(job.upload_name).stem
        job.progress("download", 1)

        segments = core.transcribe(job.source, lambda f: job.progress("transcribe", f), job.note)
        if not segments:
            raise RuntimeError("No speech found in this video, so there's nothing to clip.")
        job.note(f"Transcript ready: {len(segments)} lines")
        job.progress("transcribe", 1)

        duration = core.probe(job.source)[3]
        job.note("Picking the best moments" + (" with Claude" if job.use_llm else ""))
        job.progress("pick", 0)
        clips, method = picker.pick(segments, duration, title, job.use_llm)
        job.note(f"Picked {len(clips)} clip{'s' if len(clips) != 1 else ''} using {method}")
        job.progress("pick", 1)

        words = [w for s in segments for w in s["words"]]
        outdir = job.dir / "reels"
        outdir.mkdir(exist_ok=True)
        tasks = [(i, c, lay) for i, c in enumerate(clips, 1) for lay in job.layouts]
        for n, (i, c, lay) in enumerate(tasks):
            name = f"{i:02d}_{core.slug(c['title'])}_{lay}"
            job.note(f"Rendering clip {i} of {len(clips)} ({'face tracking' if lay == 'tracked' else 'padded'}"
                     f"{', GPU encoder' if core.encoder() == 'nvenc' else ''})")
            real = core.render_clip(
                job.source, c["start"], c["end"], lay, job.captions, words, outdir, name,
                lambda f, n=n: job.progress("render", (n + f) / len(tasks)))
            job.clips.append({
                "file": f"{name}.mp4", "index": i, "title": c["title"], "caption": c["caption"],
                "layout": lay, "duration": round(real, 1), "start": round(c["start"], 1),
                "end": round(c["end"], 1), "size": (outdir / f"{name}.mp4").stat().st_size,
            })
        write_posts(job, outdir)
        job.source.unlink(missing_ok=True)  # the original is no longer needed
        job.percent = 100
        job.status = "done"
        job.note(f"Done: {len(job.clips)} reel{'s' if len(job.clips) != 1 else ''} ready")
    except core.Cancelled:
        job.status = "cancelled"
    except Exception as e:
        log.exception("job %s failed", job.id)
        job.status = "error"
        job.error = str(e) or e.__class__.__name__
        job.note("Something went wrong")
    finally:
        job.finished = time.time()
        if job.cancelled:
            delete_job(job.id)


def write_posts(job, outdir):
    md = ["# Reels\n"]
    for c in job.clips:
        md.append(f"## {c['file']}\n\n- Length: {c['duration']}s\n- Layout: {c['layout']}\n"
                  f"- Source: {c['start']}s - {c['end']}s\n\nCaption:\n\n{c['caption']}\n")
    (outdir / "posts.md").write_text("\n".join(md), encoding="utf-8")


def worker():
    while True:
        job = work.get()
        if not job.cancelled:
            run_job(job)
        work.task_done()


def delete_job(job_id):
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if job:
        shutil.rmtree(job.dir, ignore_errors=True)


def janitor():
    while True:
        time.sleep(600)
        now = time.time()
        for job in list(jobs.values()):
            if job.status in ("done", "error") and now - (job.finished or now) > JOB_TTL:
                log.info("expiring job %s", job.id)
                delete_job(job.id)


@app.on_event("startup")
def startup():
    # Jobs live in memory, so anything on disk from a previous run is unreachable.
    shutil.rmtree(DATA, ignore_errors=True)
    DATA.mkdir(parents=True, exist_ok=True)
    global GPU
    GPU = core.gpu_status()
    log.info("GPU: %s", GPU)
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=janitor, daemon=True).start()


def get_job(job_id):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/health")
def health():
    return {"ok": True, "llm": picker.llm_enabled(), "model": picker.CLAUDE_MODEL, "gpu": GPU}


@app.post("/api/jobs")
def create_job(
    url: str = Form(""),
    file: Optional[UploadFile] = File(None),
    tracked: bool = Form(False),
    padded: bool = Form(False),
    captions: bool = Form(False),
    use_llm: bool = Form(False),
):
    url = url.strip()
    has_file = file is not None and file.filename
    if not url and not has_file:
        raise HTTPException(400, "Paste a link or upload a video.")
    if url and not url.startswith(("http://", "https://")):
        raise HTTPException(400, "That doesn't look like a link. It should start with https://")
    layouts = [lay for lay, on in (("tracked", tracked), ("padded", padded)) if on]
    if not layouts:
        raise HTTPException(400, "Pick at least one layout: face tracking or padded.")
    job = Job(url if not has_file else "", file.filename if has_file else None, layouts, captions,
              use_llm and picker.llm_enabled())
    if has_file:
        suffix = Path(file.filename).suffix.lower() or ".mp4"
        job.source = job.dir / f"source{suffix}"
        with open(job.source, "wb") as f:
            shutil.copyfileobj(file.file, f, 1024 * 1024)
        try:
            core.probe(job.source)
        except Exception:
            shutil.rmtree(job.dir, ignore_errors=True)
            raise HTTPException(400, "That file isn't a video we can read.")
        job.note("Upload received")
    with jobs_lock:
        jobs[job.id] = job
    work.put(job)
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return get_job(job_id).public()


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, download: bool = False):
    job = get_job(job_id)
    if name not in {c["file"] for c in job.clips}:
        raise HTTPException(404, "Clip not found")
    path = job.dir / "reels" / name
    return FileResponse(path, media_type="video/mp4", filename=name if download else None)


@app.get("/api/jobs/{job_id}/zip")
def job_zip(job_id: str):
    job = get_job(job_id)
    if job.status != "done":
        raise HTTPException(409, "Clips aren't ready yet")
    zpath = job.dir / "reels.zip"
    if not zpath.exists():
        tmp = zpath.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for c in job.clips:
                z.write(job.dir / "reels" / c["file"], c["file"])
            z.write(job.dir / "reels" / "posts.md", "captions.md")
        tmp.replace(zpath)
    return FileResponse(zpath, media_type="application/zip", filename="reels.zip")


@app.delete("/api/jobs/{job_id}")
def job_delete(job_id: str):
    job = get_job(job_id)
    if job.status in ("queued", "running"):
        job.cancelled = True  # the worker deletes it when it stops
        if job.status == "queued":
            delete_job(job_id)
    else:
        delete_job(job_id)
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
