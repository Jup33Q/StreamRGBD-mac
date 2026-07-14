# StreamRGBD-mac — Phoenix Web UI

This Phoenix application provides a web dashboard and JSON API for controlling
the Python StreamDiffusion RGBD pipeline.

## Architecture

```
[Camera] --Membrane--> [FrameSink] --RGB--> [InferenceWorker] --Port--> [python/inference_worker.py]
                                               |
                                               | JPEG
                                               v
                                      [VideoStreamer] --MJPEG--> [Browser <img>]
```

- **Membrane** (`StreamdiffusionMac.CameraPipeline`) captures the local camera,
  converts frames to RGB, and forwards them to the inference worker.
- **Inference worker** (`StreamdiffusionMac.InferenceWorker`) spawns
  `python/inference_worker.py` as a CLI command via an Erlang Port. It sends
  RGB frames to Python and receives JPEG-encoded results.
- **Video streamer** (`StreamdiffusionMac.VideoStreamer`) holds the latest
  processed frame.
- **Phoenix controller** serves an MJPEG stream at `/api/stream/video` that the
  dashboard displays in a plain `<img>` tag.

## Requirements

- macOS 14+ with Apple Silicon
- Python 3.9–3.12
- FFmpeg (required by Membrane camera capture plugin)

```bash
brew install ffmpeg
```

## Quick Start

```bash
cd phx
mix setup
mix phx.server
```

Open <http://localhost:4000>.

Click **Start Engine** to spawn the Python inference worker and begin the
Membrane camera pipeline. The processed video stream appears in the
**Stream Preview** panel.

## JSON API

| Method | Path | Body |
|--------|------|------|
| `POST` | `/api/stream/start` | `{"prompt":"...", "model":"sdxs", "render-size":512, "output-size":512, "strength":0.5, "feedback":0.1, "width":640, "height":480}` |
| `POST` | `/api/stream/stop` | — |
| `POST` | `/api/stream/prompt` | `{"prompt":"..."}` |
| `GET`  | `/api/stream/status` | — |
| `GET`  | `/api/stream/video` | — |

The `/api/stream/start` response includes the spawned Python instance
identifiers:

```json
{
  "ok": true,
  "thread_id": 123456789,
  "instance_pid": 12345,
  "status": { "running": true, "mode": "camera", ... }
}
```

## IEx control

```elixir
StreamdiffusionMac.StreamRGBD.start_engine(prompt: "oil painting style, masterpiece")
StreamdiffusionMac.StreamRGBD.set_prompt("cyberpunk city, neon lights")
StreamdiffusionMac.StreamRGBD.status()
StreamdiffusionMac.StreamRGBD.stop_engine()
```

## Useful tasks

```bash
# Reset the SQLite database
mix ecto.reset

# Run tests
mix test

# Run precommit checks
mix precommit
```

## Learn more

- Official website: https://www.phoenixframework.org/
- Guides: https://hexdocs.pm/phoenix/overview.html
- Docs: https://hexdocs.pm/phoenix
