defmodule StreamdiffusionMac.StreamRGBD do
  @moduledoc """
  Orchestrator that wires together the Membrane camera pipeline, the Python
  inference worker, and the video streamer.

  It no longer talks to a Python HTTP API. Instead it:
    1. Starts `StreamdiffusionMac.InferenceWorker` (spawns `python/inference_worker.py`).
    2. Starts `StreamdiffusionMac.CameraPipeline` (Membrane camera capture).
    3. Camera frames flow into the inference worker and processed JPEG frames
       land in `StreamdiffusionMac.VideoStreamer`.
    4. The Phoenix controller serves an MJPEG stream from `VideoStreamer`.

  ## Public API

      StreamdiffusionMac.StreamRGBD.start_engine(prompt: "oil painting style")
      StreamdiffusionMac.StreamRGBD.stop_engine()
      StreamdiffusionMac.StreamRGBD.set_prompt("cyberpunk city")
      StreamdiffusionMac.StreamRGBD.status()
  """

  use GenServer

  require Logger

  alias Phoenix.PubSub

  @pubsub_topic "stream_rgbd:status"

  # ---------------------------------------------------------------------------
  # Client API
  # ---------------------------------------------------------------------------

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = opts[:name] || __MODULE__
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Check whether the environment is ready to start the engine.
  Returns a map with :ok/:error per component.
  """
  @spec check_environment() :: map()
  def check_environment() do
    GenServer.call(__MODULE__, :check_environment)
  end

  @doc """
  Start the engine: spawn the Python inference worker and the Membrane camera
  pipeline. Returns `{:ok, info}` with `instance_pid` and `instance_tid`.
  """
  @spec start_engine(keyword()) ::
          {:ok, %{instance_pid: integer() | nil, instance_tid: integer() | nil}}
          | {:error, atom() | String.t()}
  def start_engine(opts \\ []) do
    GenServer.call(__MODULE__, {:start_engine, opts}, 300_000)
  end

  @doc "Stop the engine and release all resources."
  @spec stop_engine() :: :ok | {:error, atom() | String.t()}
  def stop_engine() do
    GenServer.call(__MODULE__, :stop_engine, 15_000)
  end

  @doc "Change the active prompt (takes effect on next start)."
  @spec set_prompt(String.t()) :: :ok | {:error, any()}
  def set_prompt(prompt) when is_binary(prompt) do
    GenServer.call(__MODULE__, {:set_prompt, prompt})
  end

  @doc """
  Start the camera preview pipeline (raw camera feed, no AI inference).
  Useful for testing camera availability before starting the full engine.
  """
  @spec start_camera() :: {:ok, map()} | {:error, atom() | String.t()}
  def start_camera() do
    GenServer.call(__MODULE__, :start_camera, 120_000)
  end

  @doc "Stop the camera preview pipeline."
  @spec stop_camera() :: :ok | {:error, atom() | String.t()}
  def stop_camera() do
    GenServer.call(__MODULE__, :stop_camera, 15_000)
  end

  @doc "Run Python dependency setup (setup.sh or uv pip install)."
  @spec run_setup() :: :ok | {:error, String.t()}
  def run_setup() do
    StreamdiffusionMac.ModelConverter.run_setup()
  end

  @doc "Run CoreML model conversion."
  @spec convert_models() :: :ok | {:error, String.t()}
  def convert_models() do
    StreamdiffusionMac.ModelConverter.convert_models()
  end

  @doc "Fetch the current engine status."
  @spec status() :: {:ok, map()} | {:error, any()}
  def status() do
    GenServer.call(__MODULE__, :status)
  end

  # ---------------------------------------------------------------------------
  # Server callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(opts) do
    inference_worker =
      Keyword.get(opts, :inference_worker, StreamdiffusionMac.InferenceWorker)

    camera_pipeline =
      Keyword.get(opts, :camera_pipeline, StreamdiffusionMac.CameraPipeline)

    video_streamer =
      Keyword.get(opts, :video_streamer, StreamdiffusionMac.VideoStreamer)

    camera_preview =
      Keyword.get(opts, :camera_preview, StreamdiffusionMac.CameraPreview)

    state = %{
      inference_worker: inference_worker,
      camera_pipeline: camera_pipeline,
      camera_preview: camera_preview,
      video_streamer: video_streamer,
      pipeline_pid: nil,
      camera_running: false,
      running: false,
      start_opts: %{},
      instance_pid: nil,
      instance_tid: nil,
      prompt: "oil painting style, masterpiece, highly detailed"
    }

    if Keyword.get(opts, :auto_start, false) do
      send(self(), :auto_start)
    end

    {:ok, state}
  end

  @impl true
  def handle_call(:check_environment, _from, state) do
    python_ok = File.exists?(state.inference_worker.default_python_path())
    script_ok = File.exists?(state.inference_worker.default_script_path())

    coreml_dir =
      Path.expand("../coreml_models", state.inference_worker.default_script_path())

    models = ["unet_sdxs_512.mlpackage", "taesd_decoder.mlpackage", "taesd_encoder.mlpackage"]
    missing_models = Enum.reject(models, &File.exists?(Path.join(coreml_dir, &1)))

    body = %{
      python: python_ok,
      script: script_ok,
      coreml_dir: coreml_dir,
      models_ready: Enum.empty?(missing_models),
      missing_models: missing_models
    }

    {:reply, body, state}
  end

  def handle_call({:start_engine, _opts}, _from, %{running: true} = state) do
    {:reply, {:error, :already_running}, state}
  end

  def handle_call({:start_engine, opts}, _from, state) do
    case do_start_engine(opts, state) do
      {:ok, instance, new_state} ->
        broadcast_status(new_state)
        {:reply, {:ok, instance}, new_state}

      {:error, reason, new_state} ->
        broadcast_status(new_state)
        {:reply, {:error, reason}, new_state}
    end
  end

  def handle_call(:stop_engine, _from, %{running: false} = state) do
    {:reply, {:error, :not_running}, state}
  end

  def handle_call(:stop_engine, _from, state) do
    Logger.info("[StreamRGBD] Stopping engine...")

    if state.pipeline_pid do
      Membrane.Pipeline.terminate(state.pipeline_pid)
    end

    state.inference_worker.stop_worker()
    new_state = reset_state(state)
    broadcast_status(new_state)
    {:reply, :ok, new_state}
  end

  def handle_call(:start_camera, _from, %{camera_running: true} = state) do
    {:reply, {:error, :camera_already_running}, state}
  end

  def handle_call(:start_camera, _from, state) do
    case do_start_camera(state) do
      {:ok, new_state} ->
        broadcast_status(new_state)
        {:reply, :ok, new_state}

      {:error, reason, new_state} ->
        broadcast_status(new_state)
        {:reply, {:error, reason}, new_state}
    end
  end

  def handle_call(:stop_camera, _from, %{camera_running: false} = state) do
    {:reply, {:error, :camera_not_running}, state}
  end

  def handle_call(:stop_camera, _from, state) do
    Logger.info("[StreamRGBD] Stopping camera preview...")

    if state.pipeline_pid do
      Membrane.Pipeline.terminate(state.pipeline_pid)
    end

    state.camera_preview.stop_worker()
    new_state = %{state | pipeline_pid: nil, camera_running: false}
    broadcast_status(new_state)
    {:reply, :ok, new_state}
  end

  def handle_call({:set_prompt, prompt}, _from, state) do
    state.inference_worker.set_prompt(prompt)
    new_state = %{state | prompt: prompt}
    broadcast_status(new_state)
    {:reply, :ok, new_state}
  end

  def handle_call(:status, _from, state) do
    body = build_status(state)
    {:reply, {:ok, body}, state}
  end

  @impl true
  def handle_info(:auto_start, state) do
    case do_start_engine([], state) do
      {:ok, _instance, new_state} ->
        broadcast_status(new_state)
        {:noreply, new_state}

      {:error, _reason, new_state} ->
        broadcast_status(new_state)
        {:noreply, new_state}
    end
  end

  def handle_info({:EXIT, pid, _reason}, %{pipeline_pid: pid} = state) do
    new_state = %{state | pipeline_pid: nil, running: false}
    broadcast_status(new_state)
    {:noreply, new_state}
  end

  def handle_info(_msg, state) do
    {:noreply, state}
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp do_start_engine(opts, state) do
    opts = Map.merge(state.start_opts, Map.new(opts))

    worker_opts =
      opts
      |> Map.put("prompt", state.prompt)
      |> Map.put_new("model", "sdxs")
      |> Map.put_new("render-size", 512)
      |> Map.put_new("output-size", 512)
      |> Map.put_new("strength", 0.5)
      |> Map.put_new("feedback", 0.1)

    Logger.info("[StreamRGBD] Starting inference worker...")

    case state.inference_worker.start_worker(worker_opts) do
      {:ok, instance} ->
        Logger.info("[StreamRGBD] Inference worker ready, starting camera pipeline...")

        pipeline_opts = [
          inference_worker: state.inference_worker,
          width: Map.get(opts, "width", 640),
          height: Map.get(opts, "height", 480)
        ]

        case state.camera_pipeline.start(pipeline_opts) do
          {:ok, pipeline_pid} ->
            new_state = %{
              state
              | running: true,
                pipeline_pid: pipeline_pid,
                instance_pid: instance.instance_pid,
                instance_tid: instance.instance_tid
            }

            {:ok, instance, new_state}

          {:error, reason} ->
            Logger.error("[StreamRGBD] Camera pipeline failed: #{inspect(reason)}")
            state.inference_worker.stop_worker()
            {:error, reason, state}
        end

      {:error, reason} ->
        Logger.error("[StreamRGBD] Inference worker failed: #{inspect(reason)}")
        {:error, reason, state}
    end
  end

  defp reset_state(state) do
    %{
      state
      | pipeline_pid: nil,
        running: false,
        camera_running: false,
        instance_pid: nil,
        instance_tid: nil
    }
  end

  defp do_start_camera(state) do
    Logger.info("[StreamRGBD] Starting camera preview...")

    case state.camera_preview.start_worker() do
      {:ok, _info} ->
        Logger.info("[StreamRGBD] Camera preview worker ready, starting pipeline...")

        pipeline_opts = [
          inference_worker: state.camera_preview,
          width: 640,
          height: 480
        ]

        case state.camera_pipeline.start(pipeline_opts) do
          {:ok, pipeline_pid} ->
            new_state = %{
              state
              | camera_running: true,
                pipeline_pid: pipeline_pid
            }

            {:ok, new_state}

          {:error, reason} ->
            Logger.error("[StreamRGBD] Camera pipeline failed: #{inspect(reason)}")
            state.camera_preview.stop_worker()
            {:error, reason, state}
        end

      {:error, reason} ->
        Logger.error("[StreamRGBD] Camera preview worker failed: #{inspect(reason)}")
        {:error, reason, state}
    end
  end

  @doc false
  @spec build_status(map()) :: map()
  defp build_status(state) do
    worker_status =
      case state.inference_worker.status() do
        {:ok, s} -> s
        {:error, reason} -> %{error: inspect(reason)}
      end

    streamer_status =
      case state.video_streamer.status() do
        {:ok, s} -> s
        {:error, reason} -> %{error: inspect(reason)}
      end

    %{
      running: state.running,
      camera_running: state.camera_running,
      ready: worker_status[:ready] || false,
      busy: worker_status[:busy] || false,
      mode: "camera",
      prompt: state.prompt,
      instance_pid: state.instance_pid,
      instance_tid: state.instance_tid,
      frame_count: streamer_status[:frame_count] || 0,
      has_frame: streamer_status[:has_frame] || false,
      error: worker_status[:error]
    }
  end

  @doc false
  @spec broadcast_status(map()) :: :ok
  defp broadcast_status(state) do
    body = build_status(state)
    PubSub.broadcast(StreamdiffusionMac.PubSub, @pubsub_topic, {:status_update, body})
  end
end
