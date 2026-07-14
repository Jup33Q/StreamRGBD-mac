defmodule StreamdiffusionMac.VideoStreamerTest do
  use ExUnit.Case, async: false

  alias StreamdiffusionMac.VideoStreamer

  setup do
    # Reset state by sending a nil frame cast.
    GenServer.cast(VideoStreamer, {:frame, nil})
    :ok
  end

  test "latest_frame returns nil initially" do
    assert VideoStreamer.latest_frame() == nil
  end

  test "put_frame stores the latest frame" do
    frame = <<0xFF, 0xD8, 0xFF>> <> "jpeg data"
    :ok = VideoStreamer.put_frame(frame)

    assert VideoStreamer.latest_frame() == frame
  end

  test "status reflects frame count" do
    before = VideoStreamer.status() |> elem(1) |> Map.fetch!(:frame_count)
    :ok = VideoStreamer.put_frame("frame1")
    :ok = VideoStreamer.put_frame("frame2")

    assert {:ok, %{frame_count: count, has_frame: true}} = VideoStreamer.status()
    assert count == before + 2
  end
end
