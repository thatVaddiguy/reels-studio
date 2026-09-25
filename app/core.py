"""Video processing: download, transcribe, face-track, render 9:16 reels."""
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import numpy as np

OUT_W, OUT_H = 1080, 1920
SAMPLE_FPS = 5          # face-analysis sample rate
SMOOTH_SAMPLES = 5      # ~1s moving average
CUT_THRESHOLD = 0.55    # histogram correlation below this = hard cut
SWITCH_RATIO = 1.6      # a new face must be this much larger to steal focus
YUNET = Path(os.environ.get("YUNET_MODEL", "/app/face_detection_yunet_2023mar.onnx"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "auto")

X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]
NVENC = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"]
COMMON_OUT = ["-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
              "-movflags", "+faststart"]
_encoder = None


def encoder():
    """'nvenc' if the GPU video encoder works, else 'x264'. Checked once."""
    global _encoder
    if _encoder is None:
        try:
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2",
                            *NVENC, "-f", "null", "-"], check=True, capture_output=True, timeout=30)
            _encoder = "nvenc"
        except Exception:
            _encoder = "x264"
    return _encoder


def enc_args():
    return (NVENC if encoder() == "nvenc" else X264) + COMMON_OUT


def gpu_status():
    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda = False
    name = None
    if cuda:
        try:
            name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        except Exception:
            name = "NVIDIA GPU"
    return {"cuda": cuda, "name": name, "encoder": encoder()}


class Cancelled(Exception):
    pass


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    st = info["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return int(st["width"]), int(st["height"]), float(num) / float(den), float(info["format"]["duration"])


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"


# ---------------------------------------------------------------- download

def download(url, dest, on_progress):
    """Download with yt-dlp; returns (path, title)."""
    import yt_dlp

    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(min(d.get("downloaded_bytes", 0) / total, 1.0))

    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest / "source.%(ext)s"),
        "noplaylist": True,
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    files = sorted(p for p in dest.glob("source.*") if p.suffix not in (".part", ".ytdl"))
    if not files:
        raise RuntimeError("Download finished but no video file was produced")
    return files[0], info.get("title") or "video"


# ---------------------------------------------------------------- transcription

_models = {}
_model_lock = threading.Lock()


def _whisper(device):
    from faster_whisper import WhisperModel
    name = WHISPER_MODEL if WHISPER_MODEL != "auto" else ("medium" if device == "cuda" else "small")
    key = (name, device)
    with _model_lock:
        if key not in _models:
            _models[key] = WhisperModel(name, device=device,
                                        compute_type="float16" if device == "cuda" else "int8")
    return _models[key], name


def transcribe(src, on_progress, on_note):
    """Returns list of segments with word timings. Tries GPU, falls back to CPU."""
    def go(device):
        model, name = _whisper(device)
        on_note(f"Transcribing with Whisper {name} on {'GPU' if device == 'cuda' else 'CPU'}")
        segs, info = model.transcribe(str(src), word_timestamps=True, vad_filter=True)
        out = []
        for s in segs:
            out.append({
                "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                          for w in (s.words or [])],
            })
            if info.duration:
                on_progress(min(s.end / info.duration, 1.0))
        return out

    try:
        return go("cuda")
    except Cancelled:
        raise
    except Exception as e:
        on_note(f"GPU unavailable ({str(e).splitlines()[0][:80]}), using CPU. This is slower.")
        return go("cpu")


# ---------------------------------------------------------------- face tracking

def analyze(src, start, dur, W, H, on_progress):
    """Sample frames at SAMPLE_FPS; return (face center x in source px or None, cut flags)."""
    import cv2
    aw = 480
    ah = int(round(H * aw / W / 2)) * 2
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (aw, ah), 0.7)
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", str(src),
         "-vf", f"fps={SAMPLE_FPS},scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    xs, cuts, prev_hist, prev_x = [], [], None, None
    fsize = aw * ah * 3
    expected = max(1, dur * SAMPLE_FPS)
    try:
        while True:
            buf = p.stdout.read(fsize)
            if len(buf) < fsize:
                break
            f = np.frombuffer(buf, np.uint8).reshape(ah, aw, 3)
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            cuts.append(prev_hist is not None and
                        cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < CUT_THRESHOLD)
            prev_hist = hist
            _, faces = detector.detect(np.ascontiguousarray(f))
            faces = [r for r in (faces if faces is not None else []) if r[3] >= ah / 12]
            if faces:
                # Stick with the current subject unless another face is clearly bigger.
                best = max(faces, key=lambda r: r[2] * r[3])
                if prev_x is not None and not cuts[-1]:
                    near = min(faces, key=lambda r: abs(r[0] + r[2] / 2 - prev_x))
                    if best[2] * best[3] < SWITCH_RATIO * near[2] * near[3]:
                        best = near
                x, y, w, h = best[:4]
                prev_x = x + w / 2
                xs.append(prev_x * W / aw)
            else:
                xs.append(None)
            if len(xs) % 10 == 0:
                on_progress(min(len(xs) / expected, 1.0))
    finally:
        p.kill()
        p.wait()
    return xs, cuts


def smooth_track(xs, cuts, W, crop_w):
    """Fill gaps and smooth within each scene; scenes snap at cuts."""
    n = len(xs)
    if n == 0:
        return np.array([W / 2]), np.zeros(1, int)
    scenes, s = [], 0
    for i in range(1, n):
        if cuts[i]:
            scenes.append((s, i))
            s = i
    scenes.append((s, n))
    out = np.zeros(n)
    scene_id = np.zeros(n, int)
    last = W / 2
    for k, (a, b) in enumerate(scenes):
        seg = xs[a:b]
        known = [v for v in seg if v is not None]
        if not known:
            vals = np.full(b - a, last)
        else:
            vals, cur = [], known[0]
            for v in seg:
                if v is not None:
                    cur = v
                vals.append(cur)
            vals = np.array(vals, float)
            if len(vals) > 1:
                pad = SMOOTH_SAMPLES // 2
                ext = np.pad(vals, pad, mode="edge")
                vals = np.convolve(ext, np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES, mode="valid")[:b - a]
        out[a:b] = vals
        scene_id[a:b] = k
        last = vals[-1]
    half = crop_w / 2
    return np.clip(out, half, W - half), scene_id


def _render_tracked(src, start, dur, W, H, fps, track, scene_id, vf_post, out, cwd, on_progress):
    import cv2
    crop_w = int(H * 9 / 16) // 2 * 2
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", f"{fps}", "-i", "-",
         "-ss", f"{start}", "-t", f"{dur}", "-i", str(src),
         "-map", "0:v", "-map", "1:a?", *(["-vf", vf_post] if vf_post else []),
         *enc_args(), "-shortest", str(out)],
        stdin=subprocess.PIPE, cwd=cwd)
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    fsize = W * H * 3
    n, i = len(track), 0
    total = max(1, dur * fps)
    ok = False
    try:
        while True:
            buf = dec.stdout.read(fsize)
            if len(buf) < fsize:
                break
            t = i / fps * SAMPLE_FPS
            j = min(int(t), n - 1)
            k = min(j + 1, n - 1)
            frac = t - int(t)
            cx = track[j] + (track[k] - track[j]) * frac if scene_id[j] == scene_id[k] else track[j]
            x0 = max(0, min(W - crop_w, int(round(cx - crop_w / 2))))
            frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)[:, x0:x0 + crop_w]
            enc.stdin.write(cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4).tobytes())
            i += 1
            if i % 15 == 0:
                on_progress(min(i / total, 1.0))
        enc.stdin.close()
        ok = enc.wait() == 0
    finally:
        dec.kill()
        dec.wait()
        if not ok:
            enc.kill()
            enc.wait()
    if not ok:
        raise RuntimeError("ffmpeg failed while encoding the tracked clip")


def _run_ffmpeg(cmd, dur, cwd, on_progress):
    """Run ffmpeg reporting progress via -progress."""
    p = subprocess.Popen(cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    ok = False
    try:
        for line in p.stdout:
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    us = int(line.split("=", 1)[1])
                    on_progress(min(us / 1e6 / max(dur, 0.1), 1.0))
                except ValueError:
                    pass
        err = p.stderr.read()
        ok = p.wait() == 0
    finally:
        if not ok:
            p.kill()
            p.wait()
    if not ok:
        raise RuntimeError(f"ffmpeg failed: {err.strip()[-300:]}")


# ---------------------------------------------------------------- captions

def _ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_escape(s):
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def write_ass(words, start, end, path, margin_v):
    ws = [dict(w, start=max(w["start"], start) - start, end=min(w["end"], end) - start)
          for w in words if w["end"] > start and w["start"] < end and w["word"]]
    chunks, cur = [], []
    for w in ws:
        if cur and (len(cur) >= 3 or w["start"] - cur[-1]["end"] > 0.6
                    or w["end"] - cur[0]["start"] > 1.6 or cur[-1]["word"][-1:] in ".?!,"):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {OUT_W}", f"PlayResY: {OUT_H}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,DejaVu Sans,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,2,2,80,80,{margin_v},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for ci, ch in enumerate(chunks):
        nxt = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else end - start
        ch_end = min(ch[-1]["end"] + 0.35, nxt)
        for wi, w in enumerate(ch):
            t0 = w["start"] if wi else ch[0]["start"]
            t1 = ch[wi + 1]["start"] if wi + 1 < len(ch) else ch_end
            if t1 <= t0:
                continue
            txt = " ".join(("{\\c&H00E5FF&}" + _ass_escape(x["word"].upper()) + "{\\c&HFFFFFF&}")
                           if xi == wi else _ass_escape(x["word"].upper()) for xi, x in enumerate(ch))
            lines.append(f"Dialogue: 0,{_ass_time(t0)},{_ass_time(t1)},Cap,,0,0,0,,{txt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- render

def render_clip(src, start, end, layout, captions, words, outdir, name, on_progress):
    """Render one clip. layout: 'tracked' or 'padded'. Returns output duration in seconds."""
    W, H, fps, total = probe(src)
    end = min(end, total)
    dur = end - start
    out = f"{name}.mp4"
    if layout == "tracked" and W <= H * 9 / 16:
        layout = "padded"  # source is already vertical; nothing to crop
    ass_name = f"{name}.ass"
    vf_caps = None
    if captions and words:
        write_ass(words, start, end, outdir / ass_name, 520 if layout == "padded" else 420)
        vf_caps = f"ass={ass_name}"
    try:
        if layout == "tracked":
            xs, cuts = analyze(src, start, dur, W, H, lambda f: on_progress(0.3 * f))
            crop_w = int(H * 9 / 16) // 2 * 2
            track, sid = smooth_track(xs, cuts, W, crop_w)
            _render_tracked(src, start, dur, W, H, fps, track, sid, vf_caps, out, outdir,
                            lambda f: on_progress(0.3 + 0.7 * f))
        else:
            fc = (f"[0:v]split[a][b];"
                  f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
                  f"gblur=sigma=40,eq=brightness=-0.08[bg];"
                  f"[b]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
                  f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1" + (f",{vf_caps}" if vf_caps else "") + "[v]")
            _run_ffmpeg(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", str(src),
                         "-filter_complex", fc, "-map", "[v]", "-map", "0:a?", *enc_args(), out],
                        dur, outdir, on_progress)
    finally:
        (outdir / ass_name).unlink(missing_ok=True)
    return probe(outdir / out)[3]
