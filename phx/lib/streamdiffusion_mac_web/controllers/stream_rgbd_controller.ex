defmodule StreamdiffusionMacWeb.StreamRGBDController do
  @moduledoc """
  JSON controller that exposes the StreamRGBD engine controls over HTTP.

  Endpoints delegate to `StreamdiffusionMac.StreamRGBD`.
  """

  use StreamdiffusionMacWeb, :controller

  def start(conn, params) do
    opts =
      Map.take(params, [
        "prompt",
        "model",
        "render-size",
        "output-size",
        "strength",
        "feedback",
        "width",
        "height"
      ])

    case StreamdiffusionMac.StreamRGBD.start_engine(opts) do
      {:ok, instance} ->
        json(conn, %{
          ok: true,
          thread_id: instance.instance_tid,
          instance_pid: instance.instance_pid,
          status: status_body()
        })

      {:error, reason} ->
        conn
        |> put_status(503)
        |> json(%{ok: false, error: inspect(reason)})
    end
  end

  def stop(conn, _params) do
    case StreamdiffusionMac.StreamRGBD.stop_engine() do
      :ok ->
        json(conn, %{ok: true, status: status_body()})

      {:error, reason} ->
        conn
        |> put_status(503)
        |> json(%{ok: false, error: inspect(reason)})
    end
  end

  def prompt(conn, %{"prompt" => prompt}) do
    case StreamdiffusionMac.StreamRGBD.set_prompt(prompt) do
      :ok -> json(conn, %{ok: true, status: status_body()})
      {:error, reason} -> respond_error(conn, reason)
    end
  end

  def prompt(conn, _params) do
    conn
    |> put_status(400)
    |> json(%{ok: false, error: "missing prompt"})
  end

  def status(conn, _params) do
    respond(conn, StreamdiffusionMac.StreamRGBD.status())
  end

  def video(conn, _params) do
    conn
    |> put_resp_header("content-type", "multipart/x-mixed-replace; boundary=frame")
    |> put_resp_header("cache-control", "no-cache")
    |> send_chunked(200)
    |> stream_mjpeg()
  end

  defp stream_mjpeg(conn) do
    frame = StreamdiffusionMac.VideoStreamer.latest_frame()

    {conn, ok?} =
      if frame do
        chunk = ["--frame\r\n", "Content-Type: image/jpeg\r\n\r\n", frame, "\r\n"]

        case Plug.Conn.chunk(conn, chunk) do
          {:ok, conn} -> {conn, true}
          {:error, _reason} -> {conn, false}
        end
      else
        {conn, true}
      end

    if ok? do
      Process.sleep(33)
      stream_mjpeg(conn)
    else
      conn
    end
  end

  def setup(conn, _params) do
    Task.start(fn -> StreamdiffusionMac.StreamRGBD.run_setup() end)
    json(conn, %{ok: true, message: "Setup started. Check terminal for progress."})
  end

  def convert(conn, _params) do
    Task.start(fn -> StreamdiffusionMac.StreamRGBD.convert_models() end)
    json(conn, %{ok: true, message: "Model conversion started. Check terminal for progress."})
  end

  defp respond(conn, {:ok, body}) do
    json(conn, %{ok: true, status: body})
  end

  defp respond(conn, {:error, reason}) do
    respond_error(conn, reason)
  end

  defp respond_error(conn, reason) do
    conn
    |> put_status(503)
    |> json(%{ok: false, error: inspect(reason)})
  end

  defp status_body do
    case StreamdiffusionMac.StreamRGBD.status() do
      {:ok, body} -> body
      {:error, reason} -> %{error: inspect(reason)}
    end
  end
end
