defmodule StreamdiffusionMac.CameraPipeline do
  @moduledoc """
  Membrane pipeline that captures the local camera, converts frames to RGB,
  and forwards them to a `StreamdiffusionMac.Membrane.FrameSink`.

  The sink then pushes frames into `StreamdiffusionMac.InferenceWorker`.
  """

  use Membrane.Pipeline

  alias StreamdiffusionMac.Membrane.FrameSink

  @default_width 640
  @default_height 480

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    Membrane.Pipeline.start_link(__MODULE__, opts)
  end

  @spec start(keyword()) :: GenServer.on_start()
  def start(opts \\ []) do
    Membrane.Pipeline.start(__MODULE__, opts)
  end

  @impl true
  def handle_init(_ctx, opts) do
    inference_worker = opts[:inference_worker] || StreamdiffusionMac.InferenceWorker
    width = opts[:width] || @default_width
    height = opts[:height] || @default_height

    # Membrane.CameraCapture pushes frames in the camera's native format
    # (typically I420). We convert to RGB and resize to a resolution that the
    # Python worker can handle efficiently.
    spec =
      child(:camera, Membrane.CameraCapture)
      |> child(:converter, %Membrane.FFmpeg.SWScale.Converter{
        format: :RGB,
        output_width: width,
        output_height: height
      })
      |> child(:sink, %FrameSink{consumer: inference_worker})

    {[spec: spec], %{}}
  end
end
