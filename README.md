# Reels Studio

Paste a link or upload a video and get vertical 9:16 clips with optional face tracking, padding and captions.

## Start / stop

```powershell
.\start.ps1   # builds if needed, starts the container, opens http://localhost:8000
.\stop.ps1
```

The NVIDIA GPU is used by default for transcription and video encoding. On a machine without one, `start.ps1` adds `docker-compose.cpu.yml` automatically. The footer of the page shows whether the GPU is active.

## Settings

Copy `.env.example` to `.env`:

| Variable | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Enables Claude for picking clips and writing captions. Without it, a basic picker is used. |
| `CLAUDE_MODEL` | `claude-opus-5` | Use `claude-haiku-4-5` for cheaper picks. |
| `WHISPER_MODEL` | `auto` | `medium` on GPU, `small` on CPU. |
| `JOB_TTL_HOURS` | `24` | Deletes finished jobs nobody cleared. |

## Storage

- Clips exist only inside the container until you click **I am done**. That button, the TTL, or a container restart deletes them.
- The original video is deleted once the clips are rendered.
- Whisper models are cached in the `whisper-models` volume, so they are downloaded once.
