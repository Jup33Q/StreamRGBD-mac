defmodule StreamdiffusionMac.ModelConverter do
  @moduledoc """
  GenServer that runs long-running shell commands (setup.sh, convert_models.py)
  via Erlang Port and streams their stdout/stderr back through Phoenix PubSub.

  Provides a terminal-like experience in the LiveView dashboard.
  """

  use GenServer

  require Logger

  alias Phoenix.PubSub

  @pubsub_topic "stream_rgbd:terminal"
  @project_root Path.expand("../../../..", __DIR__)

  # ---------------------------------------------------------------------------
  # Client API
  # ---------------------------------------------------------------------------

  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = opts[:name] || __MODULE__
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Run the setup script (python/setup.sh) to install Python dependencies.
  Streams log lines through PubSub `"stream_rgbd:terminal"`.
  """
  @spec run_setup() :: :ok | {:error, String.t()}
  def run_setup() do
    GenServer.call(__MODULE__, {:run, :setup}, :infinity)
  end

  @doc """
  Run the model conversion script (python/scripts/convert_models.py).
  Streams log lines through PubSub `"stream_rgbd:terminal"`.
  """
  @spec convert_models() :: :ok | {:error, String.t()}
  def convert_models() do
    GenServer.call(__MODULE__, {:run, :convert}, :infinity)
  end

  @spec running?() :: boolean()
  def running?() do
    GenServer.call(__MODULE__, :running?)
  end

  # ---------------------------------------------------------------------------
  # Server callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    {:ok, %{port: nil, from: nil, type: nil}}
  end

  @impl true
  def handle_call({:run, type}, from, %{port: nil} = state) do
    case build_cmd(type) do
      {:ok, cmd} ->
        Logger.info("[ModelConverter] Starting: #{cmd}")
        broadcast({:terminal, :clear, nil})
        broadcast({:terminal, :line, "▶ #{cmd}\n"})
        broadcast({:terminal, :running, true})

        port =
          Port.open({:spawn, String.to_charlist(cmd)}, [
            :binary,
            :exit_status,
            :stderr_to_stdout,
            :line,
            line_length: 4096
          ])

        {:noreply, %{state | port: port, from: from, type: type}}

      {:error, msg} ->
        {:reply, {:error, msg}, state}
    end
  end

  def handle_call({:run, _type}, _from, state) do
    {:reply, {:error, "Already running #{state.type}"}, state}
  end

  def handle_call(:running?, _from, state) do
    {:reply, state.port != nil, state}
  end

  @impl true
  def handle_info({port, {:data, {:eol, line}}}, %{port: port} = state) do
    broadcast({:terminal, :line, line <> "\n"})
    {:noreply, state}
  end

  def handle_info({port, {:data, {:noeol, data}}}, %{port: port} = state) do
    broadcast({:terminal, :line, data})
    {:noreply, state}
  end

  def handle_info({port, {:exit_status, status}}, %{port: port, from: from, type: type} = state) do
    if status == 0 do
      broadcast({:terminal, :line, "\n✅ #{type} completed successfully.\n"})
      broadcast({:terminal, :done, type})
      GenServer.reply(from, :ok)
    else
      broadcast({:terminal, :line, "\n❌ #{type} exited with code #{status}.\n"})
      broadcast({:terminal, :error, type})
      GenServer.reply(from, {:error, "Exit code #{status}"})
    end

    broadcast({:terminal, :running, false})
    {:noreply, %{state | port: nil, from: nil, type: nil}}
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  @uv_candidate_paths [
    "/Applications/Kimi.app/Contents/Resources/resources/daimon-bundle/runtime/uv/uv",
    "/opt/homebrew/bin/uv",
    "/usr/local/bin/uv",
    "/usr/bin/uv"
  ]

  defp find_uv do
    case System.find_executable("uv") do
      nil -> Enum.find(@uv_candidate_paths, &File.exists?/1)
      path -> path
    end
  end

  defp build_cmd(:setup) do
    uv_path = find_uv()

    if uv_path do
      cmd =
        "export PATH=\"/usr/bin:/bin:/usr/sbin:/sbin:#{Path.dirname(uv_path)}:$PATH\" && " <>
          "cd '#{@project_root}' && " <>
          "#{uv_path} pip install -r python/requirements.txt"

      {:ok, cmd}
    else
      setup_script = Path.join(@project_root, "python/setup.sh")

      if File.exists?(setup_script) do
        cmd = "cd '#{@project_root}' && bash '#{setup_script}'"
        {:ok, cmd}
      else
        {:error, "setup.sh not found at #{setup_script}"}
      end
    end
  end

  defp build_cmd(:convert) do
    uv_path = find_uv()

    python_cmd =
      if uv_path do
        "#{uv_path} run python"
      else
        venv_python = Path.join(@project_root, ".venv/bin/python")

        if File.exists?(venv_python) do
          "#{venv_python}"
        else
          "python3"
        end
      end

    cmd =
      "export PATH=\"/usr/bin:/bin:/usr/sbin:/sbin:#{if uv_path, do: Path.dirname(uv_path), else: "/usr/local/bin"}:$PATH\" && " <>
        "cd '#{@project_root}' && " <>
        "#{python_cmd} python/scripts/convert_models.py"

    {:ok, cmd}
  end

  defp broadcast(msg) do
    PubSub.broadcast(StreamdiffusionMac.PubSub, @pubsub_topic, msg)
  end
end
