defmodule StreamdiffusionMacWeb.PageController do
  use StreamdiffusionMacWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
