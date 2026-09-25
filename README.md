# Reels Studio

**Turn any long video into vertical clips ready for Instagram Reels, TikTok and YouTube Shorts.**

Paste a YouTube or Twitch link, or upload a file. Choose how the clips should look and click **Start**. Reels Studio downloads the video, transcribes it, picks the best moments, and renders 1080×1920 clips. You can then preview them, copy their captions, and download them.

Everything runs locally in one Docker container. It uses your NVIDIA GPU when one is available.

---

## Features

- **Link or upload:** YouTube, Twitch VODs and [most other sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) via yt-dlp, or drag and drop a local file.
- **Face tracking:** crops in and follows the speaker. The crop stays on the current subject instead of jumping between faces, is smoothed over about 1s, and snaps rather than pans on hard scene cuts.
- **Padded layout:** the full frame on a blurred copy of itself. Suits wide shots, groups and product showcases.
- **Word-by-word captions:** bold burned-in subtitles that highlight the word being spoken.
- **AI clip picking (optional):** Claude reads the transcript, picks self-contained moments that open on a strong hook and end on a complete thought, and writes a caption with hashtags for each one. Turn it off to use a fast built-in picker with no API cost.
- **Live progress:** a step tracker, a progress bar and the latest updates. The page can be refreshed without losing the job.
- **Easy download:** download clips one at a time, or all at once as a zip that includes the captions.
- **Cleans up after itself:** the **I am done** button deletes a job's files from the container.
- **GPU accelerated:** Whisper transcription on CUDA and H.264 encoding on NVENC. Both fall back to the CPU automatically.

## Quick start

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/), or Docker Engine with Compose v2
- Optional: an NVIDIA GPU with a current driver. On Windows this works through Docker Desktop's WSL 2 backend. On Linux, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Optional: an [Anthropic API key](https://console.anthropic.com/) for AI clip picking

### Windows

```powershell
git clone https://github.com/thatVaddiguy/reels-studio.git
cd reels-studio
.\start.ps1
```

`start.ps1` starts Docker Desktop if it isn't running, builds the image, starts the container and opens **http://localhost:8000**. Run `.\stop.ps1` to shut it down.

### macOS / Linux

```bash
git clone https://github.com/thatVaddiguy/reels-studio.git
cd reels-studio

# With an NVIDIA GPU
docker compose up -d --build

# Without an NVIDIA GPU (including Apple Silicon)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

Then open **http://localhost:8000**.

The first build downloads a few GB of dependencies. The first job also downloads the Whisper model, which is cached in a Docker volume for later runs.

## Using it

1. **Paste a link** or switch to **Upload video**.
2. Choose the output. Each layout you select produces its own version of every clip.
   - **Face tracking**
   - **Padded**
   - **Captions**
   - **All of the above**
3. Leave **Use AI to pick the best moments** on for Claude-picked clips and captions, or turn it off to use the built-in picker.
4. Click **Start** and watch the progress.
5. Preview each clip, copy its caption, and **Download** it, or use **Download all**.
6. Click **I am done** to delete the clips from the server.

The footer shows whether the GPU is in use.

## Configuration

Copy `.env.example` to `.env` and edit it. Restart the container after changing it.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | Enables the **Use AI** option. Without it, the checkbox is disabled and the built-in picker is used. |
| `CLAUDE_MODEL` | `claude-opus-5` | Model used for clip picking. `claude-haiku-4-5` is much cheaper. |
| `WHISPER_MODEL` | `auto` | `auto` uses `medium` on GPU and `small` on CPU. Also accepts `tiny`, `base`, `small`, `medium`, `large-v3`. |
| `JOB_TTL_HOURS` | `24` | Finished jobs that nobody cleared are deleted after this many hours. |

AI picking sends only the transcript text to the Anthropic API, never the video, and is billed to your Anthropic account. It typically costs a few cents per video. If the API call fails, the job continues with the built-in picker.

## How it works

```
 link / upload
      │
      ▼
 ┌──────────┐   ┌────────────────┐   ┌──────────────┐   ┌──────────────────────────┐
 │ yt-dlp   │──►│ faster-whisper │──►│ clip picker  │──►│ render (per clip/layout) │
 │ download │   │ transcript +   │   │ Claude or    │   │ YuNet face tracking      │
 └──────────┘   │ word timings   │   │ heuristic    │   │ ffmpeg + libass captions │
                └────────────────┘   └──────────────┘   │ NVENC / x264 encoding    │
                                                        └──────────────────────────┘
```

- **Transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with word-level timestamps and voice-activity filtering.
- **Clip picking:** Claude receives the timestamped transcript with the picking rules and must answer in a strict JSON format. Without a key, the built-in picker scores 30–60s windows by speech density and energy and keeps the best ones that don't overlap.
- **Face tracking:** OpenCV's [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) detector samples 5 frames per second. Positions are smoothed within each scene, and scene cuts are detected with colour histograms.
- **Output:** 1080×1920 H.264 video with AAC audio, with `faststart` set so it plays before fully downloading.

## Project layout

```
app/
  server.py        FastAPI app: job queue, progress, downloads, cleanup
  core.py          download, transcription, face tracking, rendering
  picker.py        Claude and heuristic clip picking
  static/index.html  the web page (no build step)
Dockerfile
docker-compose.yml       GPU enabled by default
docker-compose.cpu.yml   override for machines without an NVIDIA GPU
start.ps1 / stop.ps1     Windows helpers
```

## API

The page is a thin client over a small JSON API:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | GPU status and whether AI picking is available |
| `POST` | `/api/jobs` | Start a job. Multipart fields: `url` or `file`, and `tracked`, `padded`, `captions`, `use_llm` (booleans) |
| `GET` | `/api/jobs/{id}` | Status, progress, recent updates and finished clips |
| `GET` | `/api/jobs/{id}/files/{name}` | Stream a clip. Add `?download=true` to download it as a file |
| `GET` | `/api/jobs/{id}/zip` | All clips plus `captions.md` as a zip |
| `DELETE` | `/api/jobs/{id}` | Cancel a job, or delete a finished one |

## Storage and cleanup

- Job files live only inside the container, under `/data/jobs`.
- The source video is deleted as soon as the clips are rendered.
- **I am done**, the job TTL, or a container restart deletes the clips.
- Only Whisper models are kept between restarts, in the `whisper-models` volume.
- Jobs run one at a time, so a second job waits in a queue.

## Troubleshooting

| Problem | Fix |
|---|---|
| Footer says **No GPU detected** | Check that `nvidia-smi` works on the host. On Windows, update the NVIDIA driver and use Docker Desktop's WSL 2 backend. On Linux, install the NVIDIA Container Toolkit. |
| `could not select device driver "nvidia"` | The machine has no NVIDIA GPU runtime. Start with `docker-compose.cpu.yml` as shown above. |
| YouTube download fails | yt-dlp changes often. Rebuild to get the latest version: `docker compose build --no-cache && docker compose up -d`. |
| **Use AI** is greyed out | Add `ANTHROPIC_API_KEY` to `.env`, then restart the container. |
| Face tracking frames a clip badly (people far apart, shots with no one on camera) | Use the **Padded** layout for that video. |

## Responsible use

Only clip and repost videos you own or have permission to use. Platforms may mute or remove reposted content.
