defmodule StreamdiffusionMacWeb.StreamRGBDController do
  @moduledoc """
  JSON controller that exposes the StreamRGBD engine controls over HTTP.

  All endpoints return JSON and delegate to `StreamdiffusionMac.StreamRGBD`.
  """

  use StreamdiffusionMacWeb, :controller

  def start(conn, params) do
    opts = Map.take(params, ["mode", "prompt", "ndi_source", "ndi_output", "depth"])

    case StreamdiffusionMac.StreamRGBD.start_engine(opts) do
      :ok ->
        json(conn, %{ok: true, status: status_body()})

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
    respond(conn, StreamdiffusionMac.StreamRGBD.set_prompt(prompt))
  end

  def prompt(conn, _params) do
    conn
    |> put_status(400)
    |> json(%{ok: false, error: "missing prompt"})
  end

  def input_mode(conn, %{"mode" => mode}) do
    respond(conn, StreamdiffusionMac.StreamRGBD.set_input_mode(mode))
  end

  def input_mode(conn, _params) do
    conn
    |> put_status(400)
    |> json(%{ok: false, error: "missing mode"})
  end

  def ndi_input(conn, %{"source" => source}) do
    respond(conn, StreamdiffusionMac.StreamRGBD.set_ndi_input(source))
  end

  def ndi_input(conn, _params) do
    conn
    |> put_status(400)
    |> json(%{ok: false, error: "missing source"})
  end

  def ndi_output(conn, %{"name" => name}) do
    respond(conn, StreamdiffusionMac.StreamRGBD.set_ndi_output(name))
  end

  def ndi_output(conn, _params) do
    conn
    |> put_status(400)
    |> json(%{ok: false, error: "missing name"})
  end

  def status(conn, _params) do
    respond(conn, StreamdiffusionMac.StreamRGBD.status())
  end

  defp respond(conn, {:ok, body}) do
    json(conn, %{ok: true, status: body})
  end

  defp respond(conn, {:error, reason}) do
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
