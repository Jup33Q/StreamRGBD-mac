defmodule StreamdiffusionMacWeb.Router do
  use StreamdiffusionMacWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {StreamdiffusionMacWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/", StreamdiffusionMacWeb do
    pipe_through :browser

    live "/", StreamRGBDLive
  end

  scope "/api/stream", StreamdiffusionMacWeb do
    pipe_through :api

    post "/start", StreamRGBDController, :start
    post "/stop", StreamRGBDController, :stop
    post "/prompt", StreamRGBDController, :prompt
    get "/status", StreamRGBDController, :status
    get "/video", StreamRGBDController, :video
    post "/setup", StreamRGBDController, :setup
    post "/convert", StreamRGBDController, :convert
  end
end
