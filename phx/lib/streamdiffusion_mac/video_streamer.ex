defmodule StreamdiffusionMac.VideoStreamer do
  @moduledoc """
  GenServer that holds the latest JPEG frame produced by the inference worker
  and serves it to MJPEG stream consumers.

  Supports both regular RGB frames and optional depth frames for RGBD output.
  """

  use GenServer

  require Logger

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = opts[:name] || __MODULE__
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Store a new JPEG frame."
  @spec put_frame(binary()) :: :ok
  def put_frame(jpeg) when is_binary(jpeg) do
    GenServer.cast(__MODULE__, {:frame, jpeg})
  end

  @doc "Get the latest JPEG frame, or nil if none available."
  @spec latest_frame() :: binary() | nil
  def latest_frame() do
    GenServer.call(__MODULE__, :latest_frame)
  end

  @doc "Store a new depth JPEG frame."
  @spec put_depth_frame(binary()) :: :ok
  def put_depth_frame(jpeg) when is_binary(jpeg) do
    GenServer.cast(__MODULE__, {:depth_frame, jpeg})
  end

  @doc "Get the latest depth JPEG frame, or nil if none available."
  @spec latest_depth_frame() :: binary() | nil
  def latest_depth_frame() do
    GenServer.call(__MODULE__, :latest_depth_frame)
  end

  @doc "Fetch current stream status."
  @spec status() :: {:ok, map()}
  def status() do
    GenServer.call(__MODULE__, :status)
  end

  # ---------------------------------------------------------------------------
  # Server callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    {:ok,
     %{
       frame: nil,
       frame_count: 0,
       depth_frame: nil,
       depth_frame_count: 0,
       last_at: nil
     }}
  end

  @impl true
  def handle_cast({:frame, jpeg}, state) do
    {:noreply,
     %{
       state
       | frame: jpeg,
         frame_count: state.frame_count + 1,
         last_at: System.monotonic_time(:millisecond)
     }}
  end

  def handle_cast({:depth_frame, jpeg}, state) do
    {:noreply,
     %{
       state
       | depth_frame: jpeg,
         depth_frame_count: state.depth_frame_count + 1,
         last_at: System.monotonic_time(:millisecond)
     }}
  end

  @impl true
  def handle_call(:latest_frame, _from, state) do
    {:reply, state.frame, state}
  end

  def handle_call(:latest_depth_frame, _from, state) do
    {:reply, state.depth_frame, state}
  end

  def handle_call(:status, _from, state) do
    body = %{
      frame_count: state.frame_count,
      has_frame: state.frame != nil,
      depth_frame_count: state.depth_frame_count,
      has_depth_frame: state.depth_frame != nil,
      last_at: state.last_at
    }

    {:reply, {:ok, body}, state}
  end
end
