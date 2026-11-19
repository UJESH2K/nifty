# Put NiftySim on nifty.ujesh.in

Both options are free. Pick one.

## Option A: run it on a PC at home + Cloudflare Tunnel (easiest)

The PC must stay on (and not sleep) on trading days, plus about 15 minutes after 15:30 for the
nightly retrain. No router or port changes are needed, and your home IP stays hidden.

1. **Move ujesh.in's DNS to Cloudflare** (free): sign up at https://dash.cloudflare.com, add
   `ujesh.in`, then change the nameservers at your domain registrar to the two Cloudflare gives you.
   Takes from minutes up to a few hours.
2. **Password (optional)**: no login by default. Anyone with the link can view it (they can't
   change anything in autopilot mode). To add one later, put `NIFTYSIM_PASSWORD=...` in `.env`.
3. **Install the tunnel** (PowerShell):
   ```
   winget install --id Cloudflare.cloudflared
   cloudflared tunnel login
   cloudflared tunnel create niftysim
   cloudflared tunnel route dns niftysim nifty.ujesh.in
   ```
   Copy `deploy/cloudflared-config.yml` to `C:\Users\aadit\.cloudflared\config.yml` and put in the
   tunnel ID that `tunnel create` printed. Then, in an **Administrator** PowerShell:
   `cloudflared service install`. It now starts with Windows.
4. **Start NiftySim on its own**: Task Scheduler, then Create Task. Trigger: *At log on*.
   Action: run `start_autopilot.bat` from the project folder. Settings: *Restart if it fails*.
   In Windows power settings, set *Sleep: Never* (when plugged in).
5. Open https://nifty.ujesh.in on your father's phone or laptop, then log in.

Optional, nicer login: in Cloudflare Zero Trust, then Access, protect `nifty.ujesh.in` so only your
father's Google/email address can open it (free for up to 50 users).

## Option B: free cloud server (runs 24/7 without your PC)

1. Create an **Oracle Cloud Always Free** account and an Ubuntu VM (Ampere A1, 2 OCPU / 12 GB is
   plenty). In its network security list, allow ports 80 and 443.
2. Copy this project folder to the VM (including `data/` and `models_store/`), then
   `bash deploy/install_ubuntu.sh`.
3. In your DNS (registrar or Cloudflare), add an **A record**: `nifty` pointing to the VM's public IP.
   Caddy gets the HTTPS certificate automatically.
4. Optional password in `.env`, then `sudo systemctl restart niftysim`.

Note: NSE often blocks cloud-server IPs. The default `auto` source then switches to Yahoo by
itself, which can lag a little.

## What runs by itself every trading day

09:15 go live, then predict every minute, then save every tick, bar and headline. 15:30 score the day
(Lab, "Is it getting better?"), then retrain on all data including today, then wait for the next
open. Holidays and weekends are skipped.

Keep it private. Showing its predictions publicly as buy/sell calls would fall under SEBI's
research-analyst and investment-adviser rules.
