defmodule StreamdiffusionMac.StreamRGBD do
  @moduledoc """
  GenServer that owns the external Python StreamDiffusion API process.

  Provides functions to start/stop the Python runtime and to control the
  pipeline (prompt, input mode, NDI input/output names) via HTTP calls to
  the local API server.

  The Python process is spawned with an Erlang port. Stdout from the process
  is captured and logged; when the port closes, the GenServer marks the
  engine as stopped.

  ## Public API

      StreamdiffusionMac.StreamRGBD.start_engine()
      StreamdiffusionMac.StreamRGBD.stop_engine()
      StreamdiffusionMac.StreamRGBD.set_prompt("cyberpunk city, neon lights")
      StreamdiffusionMac.StreamRGBD.set_input_mode("ndi")
      StreamdiffusionMac.StreamRGBD.set_ndi_input("OBS")
      StreamdiffusionMac.StreamRGBD.set_ndi_output("SD-Render")
      StreamdiffusionMac.StreamRGBD.status()
  """

  use GenServer

  require Logger

  @default_api_port 8787
  @default_api_host "127.0.0.1"
  @api_ready_marker "STREAMDIFFUSION_API_READY"

  # -----------------------------------------------------------------------------
  # Client API
  # -----------------------------------------------------------------------------

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = opts[:name] || __MODULE__
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Start the Python StreamDiffusion API process."
  @spec start_engine(keyword()) :: :ok | {:error, atom() | String.t()}
  def start_engine(opts \\ []) do
    GenServer.call(__MODULE__, {:start_engine, opts}, 60_000)
  end

  @doc "Stop the Python StreamDiffusion API process."
  @spec stop_engine() :: :ok | {:error, atom() | String.t()}
  def stop_engine() do
    GenServer.call(__MODULE__, :stop_engine, 10_000)
  end

  @doc "Change the active prompt."
  @spec set_prompt(String.t()) :: {:ok, map()} | {:error, any()}
  def set_prompt(prompt) when is_binary(prompt) do
    GenServer.call(__MODULE__, {:api_post, "/prompt", %{prompt: prompt}})
  end

  @doc "Switch input mode: 'camera' or 'ndi'."
  @spec set_input_mode(String.t()) :: {:ok, map()} | {:error, any()}
  def set_input_mode(mode) when mode in ["camera", "ndi"] do
    GenServer.call(__MODULE__, {:api_post, "/input_mode", %{mode: mode}})
  end

  @doc "Set the NDI input source name (partial match)."
  @spec set_ndi_input(String.t()) :: {:ok, map()} | {:error, any()}
  def set_ndi_input(source) when is_binary(source) do
    GenServer.call(__MODULE__, {:api_post, "/ndi_input", %{source: source}})
  end

  @doc "Set the NDI output source name."
  @spec set_ndi_output(String.t()) :: {:ok, map()} | {:error, any()}
  def set_ndi_output(name) when is_binary(name) do
    GenServer.call(__MODULE__, {:api_post, "/ndi_output", %{name: name}})
  end

  @doc "Fetch the current engine status from the Python API."
  @spec status() :: {:ok, map()} | {:error, any()}
  def status() do
    GenServer.call(__MODULE__, {:api_get, "/status"})
  end

  # -----------------------------------------------------------------------------
  # Server callbacks
  # -----------------------------------------------------------------------------

  @impl true
  def init(opts) do
    state = %{
      port: nil,
      ready: false,
      host: Keyword.get(opts, :api_host, @default_api_host),
      port_no: Keyword.get(opts, :api_port, @default_api_port),
      python_path: Keyword.get(opts, :python_path, default_python_path()),
      script_path: Keyword.get(opts, :script_path, default_script_path()),
      start_opts: Keyword.get(opts, :start_opts, %{}),
      base_url: nil,
      stdout_buffer: ""
    }

    state = %{state | base_url: "http://#{state.host}:#{state.port_no}"}

    if Keyword.get(opts, :auto_start, false) do
      send(self(), :auto_start)
    end

    {:ok, state}
  end

  @impl true
  def handle_call({:start_engine, overrides}, _from, %{port: nil} = state) do
    start_opts = Map.merge(state.start_opts, Map.new(overrides))

    cmd =
      "#{state.python_path} #{state.script_path} " <>
        "--host #{state.host} --port #{state.port_no}"

    Logger.info("[StreamRGBD] Starting Python API: #{cmd}")

    port =
      Port.open({:spawn, String.to_charlist(cmd)}, [
        :binary,
        :stderr_to_stdout,
        :exit_status,
        line: 1024
      ])

    state = %{state | port: port, ready: false}

    # Wait for the ready marker before returning.
    case wait_for_ready(state, 30_000) do
      :ok ->
        Logger.info("[StreamRGBD] Python API ready at #{state.base_url}")

        # Loading CoreML models can take a while on first start.
        # skip_ready_check: the API server is up, even though the engine is not.
        case api_post(state, "/start", start_opts,
               receive_timeout: 120_000,
               skip_ready_check: true
             ) do
          {:ok, _body} ->
            {:reply, :ok, %{state | ready: true}}

          {:error, reason} ->
            Logger.error("[StreamRGBD] /start failed: #{inspect(reason)}")
            {:reply, {:error, reason}, state}
        end

      {:error, reason} ->
        Port.close(port)
        {:reply, {:error, reason}, %{state | port: nil, ready: false}}
    end
  end

  def handle_call({:start_engine, _opts}, _from, state) do
    {:reply, {:error, :already_running}, state}
  end

  def handle_call(:stop_engine, _from, %{port: nil} = state) do
    {:reply, {:error, :not_running}, state}
  end

  def handle_call(:stop_engine, _from, state) do
    # Ask the API to stop gracefully, then close the port.
    _ = api_post(state, "/stop", %{})
    Process.sleep(200)

    if state.port do
      Port.close(state.port)
    end

    {:reply, :ok, %{state | port: nil, ready: false}}
  end

  def handle_call({:api_post, path, payload}, _from, state) do
    {:reply, api_post(state, path, payload), state}
  end

  def handle_call({:api_get, path}, _from, state) do
    {:reply, api_get(state, path), state}
  end

  @impl true
  def handle_info({port, {:data, {:eol, line}}}, %{port: port} = state) do
    line = String.trim(line)
    Logger.info("[StreamRGBD] #{line}")

    ready? = state.ready or String.contains?(line, @api_ready_marker)
    {:noreply, %{state | ready: ready?}}
  end

  def handle_info({port, {:data, data}}, %{port: port} = state) when is_binary(data) do
    buffer = state.stdout_buffer <> data

    {lines, rest} =
      case String.split(buffer, "\n") do
        [single] ->
          {[], single}

        parts ->
          {last, rest} = List.pop_at(parts, -1)
          {rest, last}
      end

    Enum.each(lines, fn line ->
      Logger.info("[StreamRGBD] #{String.trim(line)}")
    end)

    {:noreply, %{state | stdout_buffer: rest}}
  end

  def handle_info({port, {:exit_status, status}}, %{port: port} = state) do
    Logger.warning("[StreamRGBD] Python process exited with status #{status}")
    {:noreply, %{state | port: nil, ready: false}}
  end

  def handle_info(:auto_start, state) do
    case handle_call({:start_engine, []}, :auto_start, state) do
      {:reply, _, new_state} -> {:noreply, new_state}
      other -> other
    end
  end

  def handle_info(msg, state) do
    Logger.debug("[StreamRGBD] unexpected message: #{inspect(msg)}")
    {:noreply, state}
  end

  # -----------------------------------------------------------------------------
  # Private helpers
  # -----------------------------------------------------------------------------

  defp default_python_path do
    # Assumes this file lives in phx/lib/streamdiffusion_mac.
    Path.join([__DIR__, "..", "..", "..", ".venv", "bin", "python"])
    |> Path.expand()
  end

  defp default_script_path do
    Path.join([__DIR__, "..", "..", "..", "python", "streamdiffusion_api.py"])
    |> Path.expand()
  end

  defp wait_for_ready(_state, timeout) when timeout <= 0 do
    {:error, :timeout_waiting_for_api}
  end

  defp wait_for_ready(state, timeout) do
    receive do
      {port, {:data, {:eol, line}}} when port == state.port ->
        line = String.trim(line)
        Logger.info("[StreamRGBD] #{line}")

        if String.contains?(line, @api_ready_marker) do
          :ok
        else
          wait_for_ready(state, timeout)
        end

      {port, {:data, data}} when port == state.port and is_binary(data) ->
        # Fragmented output before line mode kicks in.
        wait_for_ready(state, timeout)
    after
      100 ->
        wait_for_ready(state, timeout - 100)
    end
  end

  defp api_post(state, path, payload, opts \\ []) do
    if state.ready == false and not Keyword.get(opts, :skip_ready_check, false) do
      {:error, :api_not_ready}
    else
      url = state.base_url <> path
      receive_timeout = Keyword.get(opts, :receive_timeout, 10_000)

      case Req.post(url, json: payload, receive_timeout: receive_timeout) do
        {:ok, %{status: status, body: body}} when status in 200..299 ->
          {:ok, decode_body(body)}

        {:ok, %{status: status, body: body}} ->
          {:error, %{status: status, body: decode_body(body)}}

        {:error, exception} ->
          {:error, Exception.message(exception)}
      end
    end
  end

  defp api_get(%{ready: false}, _path) do
    {:error, :api_not_ready}
  end

  defp api_get(state, path) do
    url = state.base_url <> path

    case Req.get(url, receive_timeout: 10_000) do
      {:ok, %{status: status, body: body}} when status in 200..299 ->
        {:ok, decode_body(body)}

      {:ok, %{status: status, body: body}} ->
        {:error, %{status: status, body: decode_body(body)}}

      {:error, exception} ->
        {:error, Exception.message(exception)}
    end
  end

  defp decode_body(body) when is_map(body), do: body

  defp decode_body(body) when is_binary(body) do
    case Jason.decode(body) do
      {:ok, decoded} -> decoded
      {:error, _} -> %{"raw" => body}
    end
  end

  defp decode_body(body), do: %{"raw" => inspect(body)}
end
