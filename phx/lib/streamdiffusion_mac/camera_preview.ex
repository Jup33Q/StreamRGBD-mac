defmodule StreamdiffusionMac.CameraPreview do
  @moduledoc """
  GenServer that spawns a Python camera preview worker via Erlang Port.

  Receives raw RGB frames from `Membrane.FrameSink`, forwards them to Python
  for JPEG encoding, and pushes the resulting JPEG frames to
  `StreamdiffusionMac.VideoStreamer`.

  This allows viewing the raw camera feed without starting the heavy AI
  inference pipeline.

  Wire protocol (same as InferenceWorker):
    Elixir → Python: `<<width::32-little, height::32-little, rgb::binary>>`
    Python → Elixir: `<<jpeg_size::32-little, jpeg::binary>>`
  """

  use GenServer

  require Logger

  @ready_timeout_ms 60_000

  # ---------------------------------------------------------------------------
  # Client API
  # ---------------------------------------------------------------------------

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = opts[:name] || __MODULE__
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Start the camera preview worker.

  Returns `{:ok, info}` once the Python worker has signalled readiness.
  """
  @spec start_worker() :: {:ok, map()} | {:error, atom() | String.t()}
  def start_worker() do
    GenServer.call(__MODULE__, :start_worker, 120_000)
  end

  @doc "Stop the camera preview worker."
  @spec stop_worker() :: :ok | {:error, atom() | String.t()}
  def stop_worker() do
    GenServer.call(__MODULE__, :stop_worker, 10_000)
  end

  @spec status() :: {:ok, map()} | {:error, any()}
  def status() do
    GenServer.call(__MODULE__, :status)
  end

  # ---------------------------------------------------------------------------
  # Server callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(opts) do
    state = %{
      port: nil,
      ready: false,
      start_from: nil,
      start_timer: nil,
      python_path: Keyword.get(opts, :python_path, default_python_path()),
      script_path: Keyword.get(opts, :script_path, default_script_path()),
      video_streamer: Keyword.get(opts, :video_streamer, StreamdiffusionMac.VideoStreamer)
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:start_worker, from, %{port: nil, start_from: nil} = state) do
    with :ok <- check_python_path(state.python_path),
         :ok <- check_script_path(state.script_path) do
      cmd = Enum.join([state.python_path, state.script_path], " ")
      Logger.info("[CameraPreview] Starting Python worker: #{cmd}")

      port =
        Port.open({:spawn, String.to_charlist(cmd)}, [
          :binary,
          {:packet, 4},
          :exit_status
        ])

      state = %{
        state
        | port: port,
          ready: false,
          start_from: from,
          start_timer: nil
      }

      timer = Process.send_after(self(), :ready_timeout, @ready_timeout_ms)
      {:noreply, %{state | start_timer: timer}}
    else
      {:error, msg} -> {:reply, {:error, msg}, state}
    end
  end

  def handle_call(:start_worker, _from, state) do
    {:reply, {:error, :already_running}, state}
  end

  def handle_call(:stop_worker, _from, %{port: nil} = state) do
    {:reply, {:error, :not_running}, state}
  end

  def handle_call(:stop_worker, _from, state) do
    {:reply, :ok, stop_and_reset(state)}
  end

  def handle_call(:status, _from, state) do
    body = %{
      running: state.port != nil,
      ready: state.ready
    }

    {:reply, {:ok, body}, state}
  end

  @impl true
  def handle_info({port, {:data, data}}, %{port: port, start_from: from} = state)
      when from != nil do
    case data do
      <<0::32-little, _pid::32-little, _tid::32-little>> ->
        state = cancel_timer(state)
        reply_to_start(state.start_from, {:ok, %{}})
        {:noreply, %{state | start_from: nil, ready: true}}

      _other ->
        {:noreply, state}
    end
  end

  def handle_info({port, {:data, data}}, %{port: port} = state) do
    <<jpeg_size::32-little, jpeg::binary>> = data
    <<jpeg::binary-size(jpeg_size), _rest::binary>> = jpeg

    send(state.video_streamer, {:frame, jpeg})
    {:noreply, state}
  end

  def handle_info({:camera_frame, _payload, nil, nil}, state) do
    {:noreply, state}
  end

  def handle_info({:camera_frame, _payload, _width, _height}, %{ready: false} = state) do
    {:noreply, state}
  end

  def handle_info({:camera_frame, payload, width, height}, state) do
    expected_size = width * height * 3

    if byte_size(payload) == expected_size do
      Port.command(state.port, <<width::32-little, height::32-little, payload::binary>>)
    else
      Logger.warning(
        "[CameraPreview] Frame size mismatch: expected #{expected_size}, got #{byte_size(payload)}"
      )
    end

    {:noreply, state}
  end

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.warning("[CameraPreview] Python process exited with status #{status}")

    if state.start_from do
      reply_to_start(state.start_from, {:error, :python_process_exited})
    end

    {:noreply, reset_state(state)}
  end

  def handle_info(:ready_timeout, %{start_from: from} = state) when from != nil do
    Logger.error("[CameraPreview] Timeout waiting for ready marker")
    state = cancel_timer(state)
    reply_to_start(from, {:error, :timeout_waiting_for_worker})
    {:noreply, stop_and_reset(state)}
  end

  def handle_info(:ready_timeout, state) do
    {:noreply, state}
  end

  def handle_info(msg, state) do
    Logger.debug("[CameraPreview] unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp default_python_path do
    Path.join([__DIR__, "..", "..", "..", ".venv", "bin", "python"])
    |> Path.expand()
  end

  defp default_script_path do
    Path.join([__DIR__, "..", "..", "..", "python", "camera_preview.py"])
    |> Path.expand()
  end

  defp check_python_path(path) do
    if File.exists?(path) do
      :ok
    else
      Logger.error("[CameraPreview] Python interpreter not found: #{path}")
      {:error, "Python interpreter not found at #{path}"}
    end
  end

  defp check_script_path(path) do
    if File.exists?(path) do
      :ok
    else
      Logger.error("[CameraPreview] Worker script not found: #{path}")
      {:error, "Worker script not found at #{path}"}
    end
  end

  defp stop_and_reset(state) do
    if state.port do
      Port.close(state.port)
    end

    reset_state(state)
  end

  defp reset_state(state) do
    %{
      state
      | port: nil,
        ready: false,
        start_from: nil,
        start_timer: nil
    }
  end

  defp cancel_timer(state) do
    if state.start_timer do
      Process.cancel_timer(state.start_timer)
    end

    %{state | start_timer: nil}
  end

  defp reply_to_start(from, reply) when is_tuple(from) do
    GenServer.reply(from, reply)
  end

  defp reply_to_start(_from, _reply), do: :ok
end
