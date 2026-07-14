defmodule StreamdiffusionMac.Membrane.FrameSink do
  @moduledoc """
  Membrane sink that forwards each raw RGB frame to a consumer process.

  Used as the tail of the camera pipeline so that incoming camera frames can
  be fed into the Python inference worker.
  """

  use Membrane.Sink

  def_input_pad(:input,
    accepted_format: %Membrane.RawVideo{pixel_format: :RGB},
    flow_control: :auto
  )

  def_options(
    consumer: [
      spec: pid(),
      description: "PID that will receive {:camera_frame, payload, width, height} messages."
    ]
  )

  @impl true
  def handle_init(_ctx, opts) do
    {[], %{consumer: opts.consumer, width: nil, height: nil}}
  end

  @impl true
  def handle_stream_format(:input, %Membrane.RawVideo{width: w, height: h}, _ctx, state) do
    send(state.consumer, {:stream_format, w, h})
    {[], %{state | width: w, height: h}}
  end

  @impl true
  def handle_buffer(:input, buffer, _ctx, state) do
    send(state.consumer, {:camera_frame, buffer.payload, state.width, state.height})
    {[], state}
  end
end
