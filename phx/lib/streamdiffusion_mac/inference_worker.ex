defmodule StreamdiffusionMac.InferenceWorker do
  @moduledoc """
  GenServer that owns the Python inference worker Port.

  Responsibilities:
    - Spawn `python/inference_worker.py` as a CLI command via Erlang Port.
    - Accept raw RGB frames from the Membrane camera pipeline.
    - Forward frames to Python and receive JPEG-encoded results.
    - Push the latest processed frame to `StreamdiffusionMac.VideoStreamer`.
    - Expose runtime controls (prompt) and status.

  Wire protocol (Erlang Port with `:packet, 4`):
    - Elixir → Python frame: `<<width::32-little, height::32-little, rgb::binary>>`
    - Elixir → Python prompt: `<<0xFFFFFFFF::32-little, prompt_len::32-little, prompt::binary>>`
    - Python → Elixir: `<<jpeg_size::32-little, jpeg::binary>>`
  """

  use GenServer

  require Logger

  # Time allowed for the Python process to emit its ready packet.
  @ready_timeout_ms 180_000

  # ---------------------------------------------------------------------------
  # Client API
  # ---------------------------------------------------------------------------

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = opts[:name] || __MODULE__
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Start the Python inference worker.

  Returns `{:ok, info}` where `info` contains `instance_pid` and `instance_tid`
  once the worker has booted and loaded CoreML models.
  """
  @spec start_worker(keyword()) ::
          {:ok, %{instance_pid: integer() | nil, instance_tid: integer() | nil}}
          | {:error, atom() | String.t()}
  def start_worker(opts \\ []) do
    GenServer.call(__MODULE__, {:start_worker, opts}, 240_000)
  end

  @doc "Stop the Python inference worker."
  @spec stop_worker() :: :ok | {:error, atom() | String.t()}
  def stop_worker() do
    GenServer.call(__MODULE__, :stop_worker, 10_000)
  end

  @doc "Change the active prompt."
  @spec set_prompt(String.t()) :: :ok | {:error, any()}
  def set_prompt(prompt) when is_binary(prompt) do
    GenServer.call(__MODULE__, {:set_prompt, prompt})
  end

  @doc "Fetch current worker status."
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
      start_opts: %{},
      width: nil,
      height: nil,
      busy: false,
      pending_frame: nil,
      instance_pid: nil,
      instance_tid: nil,
      python_path: Keyword.get(opts, :python_path, default_python_path()),
      script_path: Keyword.get(opts, :script_path, default_script_path()),
      video_streamer: Keyword.get(opts, :video_streamer, StreamdiffusionMac.VideoStreamer)
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:start_worker, overrides}, from, %{port: nil, start_from: nil} = state) do
    start_opts = Map.merge(state.start_opts, Map.new(overrides))

    with :ok <- check_python_path(state.python_path),
         :ok <- check_script_path(state.script_path) do
      cmd = build_cmd(state, start_opts)
      Logger.info("[InferenceWorker] Starting Python worker: #{cmd}")

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
          start_opts: start_opts,
          instance_pid: nil,
          instance_tid: nil
      }

      timer = Process.send_after(self(), :ready_timeout, @ready_timeout_ms)
      {:noreply, %{state | start_timer: timer}}
    else
      {:error, msg} -> {:reply, {:error, msg}, state}
    end
  end

  def handle_call({:start_worker, _opts}, _from, state) do
    {:reply, {:error, :already_running}, state}
  end

  def handle_call(:stop_worker, _from, %{port: nil} = state) do
    {:reply, {:error, :not_running}, state}
  end

  def handle_call(:stop_worker, _from, state) do
    reply = stop_and_reset(state)
    {:reply, :ok, reply}
  end

  def handle_call({:set_prompt, _prompt}, _from, %{port: nil} = state) do
    {:reply, {:error, :not_running}, state}
  end

  def handle_call({:set_prompt, prompt}, _from, %{ready: true} = state) do
    # Send prompt update through the Port using the wire protocol sentinel.
    # Width == 0xFFFFFFFF signals a prompt update packet instead of a frame.
    send_prompt_update(state.port, prompt)
    state = %{state | start_opts: Map.put(state.start_opts, "prompt", prompt)}
    {:reply, :ok, state}
  end

  def handle_call({:set_prompt, _prompt}, _from, state) do
    {:reply, {:error, :not_ready}, state}
  end

  def handle_call(:status, _from, state) do
    body = %{
      running: state.port != nil,
      ready: state.ready,
      busy: state.busy,
      instance_pid: state.instance_pid,
      instance_tid: state.instance_tid,
      width: state.width,
      height: state.height
    }

    {:reply, {:ok, body}, state}
  end

  @impl true
  def handle_info({port, {:data, data}}, %{port: port, start_from: from} = state)
      when from != nil do
    case data do
      <<0::32-little, pid::32-little, tid::32-little>> ->
        state = cancel_timer(state)
        state = %{state | instance_pid: pid, instance_tid: tid}
        reply_to_start(state.start_from, {:ok, instance_info(state)})
        {:noreply, %{state | start_from: nil, ready: true}}

      _other ->
        # Stray data before ready packet; ignore.
        {:noreply, state}
    end
  end

  def handle_info({port, {:data, data}}, %{port: port} = state) do
    case data do
      <<0xFFFFFFFE::32-little, depth_jpeg_size::32-little, depth_jpeg::binary>> ->
        <<depth_jpeg_body::binary-size(depth_jpeg_size), _rest::binary>> = depth_jpeg
        send(state.video_streamer, {:depth_frame, depth_jpeg_body})
        {:noreply, state}

      <<jpeg_size::32-little, jpeg::binary>> ->
        <<jpeg::binary-size(jpeg_size), _rest::binary>> = jpeg
        send(state.video_streamer, {:frame, jpeg})
        state = %{state | busy: false}
        state = maybe_send_pending(state)
        {:noreply, state}
    end
  end

  def handle_info({:camera_frame, _payload, nil, nil}, state) do
    # No stream format yet; drop frame.
    {:noreply, state}
  end

  def handle_info({:camera_frame, _payload, _width, _height}, %{ready: false} = state) do
    # Worker not ready yet; drop frame.
    {:noreply, state}
  end

  def handle_info({:camera_frame, payload, width, height}, state) do
    state = %{state | width: width, height: height}

    cond do
      state.busy ->
        # Keep only the latest pending frame for low latency.
        {:noreply, %{state | pending_frame: {payload, width, height}}}

      true ->
        send_frame(state.port, width, height, payload)
        {:noreply, %{state | busy: true}}
    end
  end

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.warning("[InferenceWorker] Python process exited with status #{status}")

    if state.start_from do
      reply_to_start(state.start_from, {:error, :python_process_exited})
    end

    {:noreply, reset_state(state)}
  end

  def handle_info(:ready_timeout, %{start_from: from} = state) when from != nil do
    Logger.error("[InferenceWorker] Timeout waiting for ready marker")
    state = cancel_timer(state)
    reply_to_start(from, {:error, :timeout_waiting_for_worker})
    {:noreply, stop_and_reset(state)}
  end

  def handle_info(:ready_timeout, state) do
    {:noreply, state}
  end

  def handle_info(msg, state) do
    Logger.debug("[InferenceWorker] unexpected message: #{inspect(msg)}")
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
    Path.join([__DIR__, "..", "..", "..", "python", "inference_worker.py"])
    |> Path.expand()
  end

  defp check_python_path(path) do
    if File.exists?(path) do
      :ok
    else
      Logger.error("[InferenceWorker] Python interpreter not found: #{path}")

      {:error,
       "Python interpreter not found at #{path}. Run: cd .. && python/setup.sh (requires Python 3.9-3.12)"}
    end
  end

  defp check_script_path(path) do
    if File.exists?(path) do
      :ok
    else
      Logger.error("[InferenceWorker] Worker script not found: #{path}")
      {:error, "Worker script not found at #{path}"}
    end
  end

  defp build_cmd(state, opts) do
    args =
      [
        "--prompt",
        opts["prompt"] || "oil painting style, masterpiece, highly detailed",
        "--model",
        opts["model"] || "sdxs",
        "--render-size",
        to_string(opts["render-size"] || 512),
        "--output-size",
        to_string(opts["output-size"] || 512),
        "--strength",
        to_string(opts["strength"] || 0.5),
        "--feedback",
        to_string(opts["feedback"] || 0.1),
        "--coreml-dir",
        opts["coreml-dir"] || Path.expand("../coreml_models", state.script_path)
      ]

    args =
      if opts["depth-backend"] do
        args ++
          [
            "--depth-backend",
            opts["depth-backend"],
            "--depth-coreml-path",
            opts["depth-coreml-path"] ||
              Path.expand("../coreml_models/da3_small.mlpackage", state.script_path)
          ]
      else
        args
      end

    Enum.join([state.python_path, state.script_path | args], " ")
  end

  defp send_prompt_update(port, prompt) do
    # Prompt update packet: width=0xFFFFFFFF signals this is not a frame.
    prompt_len = byte_size(prompt)
    Port.command(port, <<0xFFFFFFFF::32-little, prompt_len::32-little, prompt::binary>>)
  end

  defp send_frame(port, width, height, payload) do
    expected_size = width * height * 3

    if byte_size(payload) == expected_size do
      Port.command(port, <<width::32-little, height::32-little, payload::binary>>)
    else
      Logger.warning(
        "[InferenceWorker] Frame size mismatch: expected #{expected_size}, got #{byte_size(payload)}"
      )
    end
  end

  defp maybe_send_pending(%{pending_frame: nil} = state), do: state

  defp maybe_send_pending(state) do
    {payload, width, height} = state.pending_frame
    send_frame(state.port, width, height, payload)
    %{state | pending_frame: nil, busy: true}
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
        start_timer: nil,
        busy: false,
        pending_frame: nil,
        instance_pid: nil,
        instance_tid: nil
    }
  end

  defp cancel_timer(state) do
    if state.start_timer do
      Process.cancel_timer(state.start_timer)
    end

    %{state | start_timer: nil}
  end

  defp instance_info(state) do
    %{
      instance_pid: state.instance_pid,
      instance_tid: state.instance_tid
    }
  end

  defp reply_to_start(from, reply) when is_tuple(from) do
    GenServer.reply(from, reply)
  end

  defp reply_to_start(_from, _reply), do: :ok
end
