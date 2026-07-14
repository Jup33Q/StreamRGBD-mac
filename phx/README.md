# StreamRGBD-mac — Phoenix Web UI

This Phoenix application provides a web dashboard and JSON API for controlling
the Python StreamDiffusion RGBD / NDI pipeline.

## Features

- **Web dashboard** at `/` — start/stop the engine, change prompt, switch
  camera/NDI input, and set NDI source/output names.
- **JSON API** under `/api/stream/*` — remote control endpoints that delegate to
  the `StreamdiffusionMac.StreamRGBD` GenServer.
- **GenServer** (`StreamdiffusionMac.StreamRGBD`) — owns the Python API process
  via an Erlang port and exposes a programmatic Elixir API.

## Quick Start

```bash
cd phx
mix setup
mix phx.server
```

Open <http://localhost:4000>.

## IEx control

```elixir
StreamdiffusionMac.StreamRGBD.start_engine()
StreamdiffusionMac.StreamRGBD.set_prompt("cyberpunk city, neon lights")
StreamdiffusionMac.StreamRGBD.set_input_mode("ndi")
StreamdiffusionMac.StreamRGBD.set_ndi_input("OBS")
StreamdiffusionMac.StreamRGBD.set_ndi_output("SD-Render")
StreamdiffusionMac.StreamRGBD.status()
StreamdiffusionMac.StreamRGBD.stop_engine()
```

## JSON API

| Method | Path | Body |
|--------|------|------|
| `POST` | `/api/stream/start` | `{"mode":"camera|ndi", "prompt":"...", "ndi_source":"...", "ndi_output":"..."}` |
| `POST` | `/api/stream/stop` | — |
| `POST` | `/api/stream/prompt` | `{"prompt":"..."}` |
| `POST` | `/api/stream/input_mode` | `{"mode":"camera|ndi"}` |
| `POST` | `/api/stream/ndi_input` | `{"source":"..."}` |
| `POST` | `/api/stream/ndi_output` | `{"name":"..."}` |
| `GET`  | `/api/stream/status` | — |

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
