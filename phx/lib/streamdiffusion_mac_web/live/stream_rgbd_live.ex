defmodule StreamdiffusionMacWeb.StreamRGBDLive do
  @moduledoc """
  LiveView for the StreamRGBD engine control panel.

  Provides real-time status updates via Phoenix.PubSub and controls for
  starting, stopping, and configuring the inference engine.
  """

  use StreamdiffusionMacWeb, :live_view

  alias StreamdiffusionMac.StreamRGBD

  @impl true
  @spec mount(map(), map(), Phoenix.LiveView.Socket.t()) :: {:ok, Phoenix.LiveView.Socket.t()}
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(StreamdiffusionMac.PubSub, "stream_rgbd:status")
    end

    {:ok, assign(socket, status: default_status())}
  end

  @impl true
  @spec handle_info({:status_update, map()}, Phoenix.LiveView.Socket.t()) ::
          {:noreply, Phoenix.LiveView.Socket.t()}
  def handle_info({:status_update, status}, socket) do
    {:noreply, assign(socket, status: status)}
  end

  @impl true
  @spec handle_event(String.t(), map(), Phoenix.LiveView.Socket.t()) ::
          {:noreply, Phoenix.LiveView.Socket.t()}
  def handle_event("start_engine", _params, socket) do
    case StreamRGBD.start_engine() do
      {:ok, _instance} ->
        {:noreply, socket}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to start engine: #{inspect(reason)}")}
    end
  end

  def handle_event("stop_engine", _params, socket) do
    StreamRGBD.stop_engine()
    {:noreply, socket}
  end

  def handle_event("set_prompt", %{"prompt" => prompt}, socket) do
    StreamRGBD.set_prompt(prompt)
    {:noreply, put_flash(socket, :info, "Prompt updated")}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="space-y-8">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">StreamDiffusion RGBD</h1>
        <p class="mt-2 text-base-content/70">
          Membrane captures the local camera, a Python CLI worker runs the
          CoreML img2img pipeline, and the processed stream is shown below.
        </p>
      </div>

      <div class="grid gap-6 md:grid-cols-2">
        <%!-- Engine lifecycle --%>
        <section class="card bg-base-200 shadow-sm">
          <div class="card-body">
            <h2 class="card-title">Engine</h2>
            <div class="flex flex-wrap gap-3 mt-2">
              <button phx-click="start_engine" class="btn btn-primary">
                Start Engine
              </button>
              <button phx-click="stop_engine" class="btn btn-secondary">
                Stop Engine
              </button>
              <a href="/api/stream/status" target="_blank" class="btn btn-outline">Status (JSON)</a>
            </div>
            <p class="text-xs text-base-content/60 mt-3">
              Starting spawns the Python inference worker via CLI and begins the
              Membrane camera pipeline.
            </p>
          </div>
        </section>

        <%!-- Live status --%>
        <section class="card bg-base-200 shadow-sm">
          <div class="card-body">
            <h2 class="card-title">Instance Status</h2>
            <dl class="grid grid-cols-2 gap-2 text-sm mt-2">
              <dt class="text-base-content/60">Running</dt>
              <dd class="font-mono">{(@status.running && "yes") || "no"}</dd>
              <dt class="text-base-content/60">Ready</dt>
              <dd class="font-mono">{(@status.ready && "yes") || "no"}</dd>
              <dt class="text-base-content/60">Busy</dt>
              <dd class="font-mono">{(@status.busy && "yes") || "no"}</dd>
              <dt class="text-base-content/60">Thread ID</dt>
              <dd class="font-mono">{@status.instance_tid || "—"}</dd>
              <dt class="text-base-content/60">Process ID</dt>
              <dd class="font-mono">{@status.instance_pid || "—"}</dd>
              <dt class="text-base-content/60">Frames Streamed</dt>
              <dd class="font-mono">{@status.frame_count || "—"}</dd>
              <dt class="text-base-content/60">Has Frame</dt>
              <dd class="font-mono">{(@status.has_frame && "yes") || "no"}</dd>
              <dt class="text-base-content/60">Error</dt>
              <dd class="font-mono">{@status.error || "—"}</dd>
            </dl>
          </div>
        </section>

        <%!-- Prompt --%>
        <section class="card bg-base-200 shadow-sm md:col-span-2">
          <div class="card-body">
            <h2 class="card-title">Prompt</h2>
            <form phx-submit="set_prompt" class="mt-2">
              <div class="join w-full">
                <input
                  type="text"
                  name="prompt"
                  value={@status.prompt}
                  placeholder="oil painting style, masterpiece"
                  class="input input-bordered join-item flex-1"
                  required
                />
                <button type="submit" class="btn btn-primary join-item">Set</button>
              </div>
            </form>
            <p class="text-xs text-base-content/60 mt-2">
              Prompt changes are sent to the Python worker in real time when the engine is ready.
            </p>
          </div>
        </section>

        <%!-- Video preview --%>
        <section class="card bg-base-200 shadow-sm md:col-span-2">
          <div class="card-body">
            <h2 class="card-title">Stream Preview</h2>
            <div class="mt-2 overflow-hidden rounded-lg bg-black aspect-video">
              <img
                src="/api/stream/video"
                alt="StreamDiffusion output"
                class="w-full h-full object-contain"
              />
            </div>
          </div>
        </section>
      </div>
    </div>
    """
  end

  defp default_status do
    %{
      running: false,
      ready: false,
      busy: false,
      mode: "camera",
      prompt: "",
      instance_pid: nil,
      instance_tid: nil,
      frame_count: 0,
      has_frame: false,
      error: nil
    }
  end
end
