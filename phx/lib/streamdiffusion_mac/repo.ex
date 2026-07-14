defmodule StreamdiffusionMac.Repo do
  use Ecto.Repo,
    otp_app: :streamdiffusion_mac,
    adapter: Ecto.Adapters.SQLite3
end
