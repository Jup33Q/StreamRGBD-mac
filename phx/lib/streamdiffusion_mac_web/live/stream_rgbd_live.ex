defmodule StreamdiffusionMacWeb.StreamRGBDLive do
  @moduledoc """
  LiveView for the StreamRGBD engine control panel.

  Provides real-time status updates via Phoenix.PubSub and controls for
  starting, stopping, and configuring the inference engine.

  Environment check on mount warns the user if CoreML models are missing.
  """

  use StreamdiffusionMacWeb, :live_view

  alias StreamdiffusionMac.StreamRGBD

  @impl true
  @spec mount(map(), map(), Phoenix.LiveView.Socket.t()) :: {:ok, Phoenix.LiveView.Socket.t()}
  def mount(_params, _session, socket) do
    env = if connected?(socket), do: StreamRGBD.check_environment(), else: %{}

    if connected?(socket) do
      Phoenix.PubSub.subscribe(StreamdiffusionMac.PubSub, "stream_rgbd:status")
      Phoenix.PubSub.subscribe(StreamdiffusionMac.PubSub, "stream_rgbd:terminal")
    end

    {:ok,
     assign(socket,
       status: default_status(),
       env: env,
       starting: false,
       terminal_log: "",
       terminal_running: false
     )}
  end

  @impl true
  @spec handle_info({:status_update, map()}, Phoenix.LiveView.Socket.t()) ::
          {:noreply, Phoenix.LiveView.Socket.t()}
  def handle_info({:status_update, status}, socket) do
    socket =
      socket
      |> assign(status: status)
      |> assign(starting: false)

    {:noreply, socket}
  end

  @impl true
  def handle_info({:terminal, :line, text}, socket) do
    new_log = socket.assigns.terminal_log <> text
    # Keep last ~500 lines to avoid memory bloat
    trimmed = trim_log(new_log)
    {:noreply, assign(socket, terminal_log: trimmed)}
  end

  def handle_info({:terminal, :clear, _}, socket) do
    {:noreply, assign(socket, terminal_log: "")}
  end

  def handle_info({:terminal, :running, running}, socket) do
    {:noreply, assign(socket, terminal_running: running)}
  end

  def handle_info({:terminal, :done, _type}, socket) do
    env = StreamRGBD.check_environment()
    {:noreply, assign(socket, env: env, terminal_running: false)}
  end

  def handle_info({:terminal, :error, _type}, socket) do
    {:noreply, assign(socket, terminal_running: false)}
  end

  @impl true
  @spec handle_event(String.t(), map(), Phoenix.LiveView.Socket.t()) ::
          {:noreply, Phoenix.LiveView.Socket.t()}
  def handle_event("start_engine", _params, socket) do
    env = socket.assigns.env

    if not env[:models_ready] do
      {:noreply,
       put_flash(
         socket,
         :error,
         "CoreML models not found. Run Setup → Convert Models first."
       )}
    else
      socket = assign(socket, starting: true)

      case StreamRGBD.start_engine() do
        {:ok, _instance} ->
          {:noreply, socket}

        {:error, reason} ->
          socket = assign(socket, starting: false)
          {:noreply, put_flash(socket, :error, "Failed to start engine: #{inspect(reason)}")}
      end
    end
  end

  def handle_event("start_camera", _params, socket) do
    socket = assign(socket, starting: true)

    case StreamRGBD.start_camera() do
      :ok ->
        {:noreply, socket}

      {:error, reason} ->
        socket = assign(socket, starting: false)
        {:noreply, put_flash(socket, :error, "Failed to start camera: #{inspect(reason)}")}
    end
  end

  def handle_event("stop_camera", _params, socket) do
    StreamRGBD.stop_camera()
    {:noreply, assign(socket, starting: false)}
  end

  def handle_event("stop_engine", _params, socket) do
    StreamRGBD.stop_engine()
    {:noreply, assign(socket, starting: false)}
  end

  def handle_event("set_prompt", %{"prompt" => prompt}, socket) do
    StreamRGBD.set_prompt(prompt)
    {:noreply, put_flash(socket, :info, "Prompt updated")}
  end

  def handle_event("check_env", _params, socket) do
    env = StreamRGBD.check_environment()
    {:noreply, assign(socket, env: env)}
  end

  def handle_event("run_setup", _params, socket) do
    Task.start(fn -> StreamRGBD.run_setup() end)
    {:noreply, socket}
  end

  def handle_event("convert_models", _params, socket) do
    Task.start(fn -> StreamRGBD.convert_models() end)
    {:noreply, socket}
  end

  def handle_event("clear_terminal", _params, socket) do
    {:noreply, assign(socket, terminal_log: "")}
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

      <%!-- Environment check banner --%>
      <%= if @env != %{} and not @env.models_ready do %>
        <div class="alert alert-warning">
          <span class="font-bold">⚠️ CoreML models missing</span>
          <div class="mt-1 text-sm">
            <p>CoreML directory: <code class="bg-base-300 px-1 rounded">{@env.coreml_dir}</code></p>
            <p>Missing models: {Enum.join(@env.missing_models, ", ")}</p>
            <p class="mt-1">
              Use the Terminal panel below to run Setup → Convert Models.
            </p>
            <button phx-click="check_env" class="btn btn-sm btn-outline mt-2">Recheck</button>
          </div>
        </div>
      <% end %>

      <%= if @env != %{} and @env.models_ready and not @status.running do %>
        <div class="alert alert-success">
          <span>✅ Environment ready. CoreML models found. Click "Start Engine" to begin.</span>
        </div>
      <% end %>

      <div class="grid gap-6 md:grid-cols-2">
        <%!-- Engine lifecycle --%>
        <section class="card bg-base-200 shadow-sm">
          <div class="card-body">
            <h2 class="card-title">Engine</h2>
            <div class="flex flex-wrap gap-3 mt-2">
              <button
                phx-click="start_engine"
                disabled={@starting or @status.running}
                class="btn btn-primary"
              >
                <%= if @starting and not @status.running do %>
                  <span class="loading loading-spinner loading-xs"></span>
                  Starting…
                <% else %>
                  Start Engine
                <% end %>
              </button>
              <button phx-click="stop_engine" disabled={not @status.running} class="btn btn-secondary">
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

        <%!-- Camera preview lifecycle --%>
        <section class="card bg-base-200 shadow-sm">
          <div class="card-body">
            <h2 class="card-title">Camera Preview</h2>
            <div class="flex flex-wrap gap-3 mt-2">
              <button
                phx-click="start_camera"
                disabled={@starting or @status.camera_running}
                class="btn btn-accent"
              >
                <%= if @starting and not @status.camera_running do %>
                  <span class="loading loading-spinner loading-xs"></span>
                  Starting…
                <% else %>
                  Open Camera
                <% end %>
              </button>
              <button phx-click="stop_camera" disabled={not @status.camera_running} class="btn btn-secondary">
                Close Camera
              </button>
            </div>
            <p class="text-xs text-base-content/60 mt-3">
              Opens the raw camera feed without AI inference.
            </p>
          </div>
        </section>

        <%!-- Live status --%>
        <section class="card bg-base-200 shadow-sm">
          <div class="card-body">
            <h2 class="card-title">Instance Status</h2>
            <dl class="grid grid-cols-2 gap-2 text-sm mt-2">
              <dt class="text-base-content/60">Engine Running</dt>
              <dd class="font-mono">{(@status.running && "yes") || "no"}</dd>
              <dt class="text-base-content/60">Camera Running</dt>
              <dd class="font-mono">{(@status.camera_running && "yes") || "no"}</dd>
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
              <dd class="font-mono text-error">{@status.error || "—"}</dd>
            </dl>
          </div>
        </section>

        <%!-- Terminal / Environment management --%>
        <section class="card bg-base-200 shadow-sm md:col-span-2">
          <div class="card-body">
            <div class="flex items-center justify-between">
              <h2 class="card-title">Terminal</h2>
              <div class="flex gap-2">
                <button
                  phx-click="run_setup"
                  disabled={@terminal_running}
                  class="btn btn-sm btn-outline"
                >
                  <%= if @terminal_running do %>
                    <span class="loading loading-spinner loading-xs"></span>
                  <% end %>
                  Setup
                </button>
                <button
                  phx-click="convert_models"
                  disabled={@terminal_running}
                  class="btn btn-sm btn-primary"
                >
                  <%= if @terminal_running do %>
                    <span class="loading loading-spinner loading-xs"></span>
                  <% end %>
                  Convert Models
                </button>
                <button phx-click="clear_terminal" class="btn btn-sm btn-ghost">Clear</button>
              </div>
            </div>
            <div class="mt-3 rounded-lg bg-neutral text-neutral-content p-3 font-mono text-xs overflow-auto" style="max-height: 300px;" id="terminal-output" phx-update="ignore">
              <pre class="whitespace-pre-wrap break-all"><%= @terminal_log %></pre>
            </div>
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
            <%= if not @status.running and not @status.camera_running do %>
              <div class="mt-2 flex items-center justify-center rounded-lg bg-black aspect-video text-white/50">
                <p>Stream stopped. Click "Start Engine" or "Open Camera" to begin.</p>
              </div>
            <% else %>
              <div class="mt-2 overflow-hidden rounded-lg bg-black aspect-video">
                <img
                  src="/api/stream/video"
                  alt="Stream output"
                  class="w-full h-full object-contain"
                />
              </div>
            <% end %>
          </div>
        </section>
      </div>
    </div>
    """
  end

  defp default_status do
    %{
      running: false,
      camera_running: false,
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

  @max_terminal_chars 50_000
  defp trim_log(log) when byte_size(log) > @max_terminal_chars do
    String.slice(log, -@max_terminal_chars, @max_terminal_chars)
  end

  defp trim_log(log), do: log
end
