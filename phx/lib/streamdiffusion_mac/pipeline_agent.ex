defmodule StreamdiffusionMac.PipelineAgent do
  @moduledoc """
  Agent that holds the runtime configuration/state for the Python
  StreamDiffusion RGBD pipeline.

  This is intentionally lightweight: it keeps settings such as the active
  prompt, blend ratio and preview mode in memory so that Phoenix controllers,
  LiveViews or scheduled jobs can read and update them without starting the
  heavy Python pipeline each time.
  """

  use Agent

  @default_state %{
    prompt: "oil painting style, masterpiece",
    model: "sdxs",
    render_size: 512,
    output_size: 512,
    blend: 0.0,
    ema: 0.4,
    feedback: 0.1,
    depth_backend: "auto",
    depth_preview_mode: "mono",
    running: false,
    last_frame: nil
  }

  @spec start_link(keyword()) :: Agent.on_start()
  def start_link(opts \\ []) do
    name = opts[:name] || __MODULE__
    initial = Keyword.get(opts, :initial_state, @default_state)
    Agent.start_link(fn -> initial end, name: name)
  end

  @doc "Return the current pipeline state."
  @spec state(atom() | pid()) :: map()
  def state(agent \\ __MODULE__) do
    Agent.get(agent, & &1)
  end

  @doc "Get a single key from the state."
  @spec get(atom() | pid(), atom()) :: any()
  def get(agent \\ __MODULE__, key) when is_atom(key) do
    Agent.get(agent, &Map.get(&1, key))
  end

  @doc "Update a single key in the state."
  @spec put(atom() | pid(), atom(), any()) :: :ok
  def put(agent \\ __MODULE__, key, value) when is_atom(key) do
    Agent.update(agent, &%{&1 | key => value})
  end

  @doc "Merge a map of values into the state."
  @spec merge(atom() | pid(), map()) :: :ok
  def merge(agent \\ __MODULE__, values) when is_map(values) do
    Agent.update(agent, &Map.merge(&1, values))
  end

  @doc "Store the latest captured frame (e.g. base64 JPEG bytes)."
  @spec put_frame(atom() | pid(), binary()) :: :ok
  def put_frame(agent \\ __MODULE__, frame_bytes) when is_binary(frame_bytes) do
    Agent.update(agent, &%{&1 | last_frame: frame_bytes})
  end

  @doc "Mark the pipeline as running or stopped."
  @spec set_running(atom() | pid(), boolean()) :: :ok
  def set_running(agent \\ __MODULE__, running) when is_boolean(running) do
    Agent.update(agent, &%{&1 | running: running})
  end
end
