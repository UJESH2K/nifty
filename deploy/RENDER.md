# NiftySim on Render (free) at nifty.ujesh.in

Render's free plan sleeps after 15 minutes without visitors and has 512 MB of memory and no lasting
disk. NiftySim handles all three:

* **Never sleeps:** the app visits its own public address every 10 minutes (and you can add a free
  outside pinger as a backup, step 5).
* **Fits in 512 MB:** the cloud copy keeps the last 320 trading days and retrains on the last 250
  (tested at about 390 MB peak). Your PC keeps all 11 years.
* **Disk resets on every restart or redeploy:** the day's ticks, the nightly retrain and the daily
  report card live only until the next restart. After a restart it starts again from what you
  pushed. Re-run step 1 and push every week or two to refresh it from your PC.

## Steps (about 20 minutes, once)

1. **Build the slim copy** (on your PC, in the project folder):
   `python scripts/make_render_bundle.py`. This creates `Downloads\niftysim\niftysim_render`.
2. **Put it on GitHub** (free account): install GitHub Desktop, then *File, Add local repository*,
   choose `niftysim_render`, *create a repository*, *Publish repository* (Private is fine).
3. **Create the service on Render** (https://render.com, sign in with GitHub): *New, Blueprint*, then
   pick the repository. It reads `render.yaml` (free plan, Singapore region) and deploys. The first
   build takes about 5 minutes; then the app is at `https://niftysim-xxxx.onrender.com`.
4. **Your domain**: in Render, open the service, then *Settings, Custom Domains, Add* and enter
   `nifty.ujesh.in`. Render shows a CNAME target. At your domain registrar's DNS page, add a
   **CNAME** record with name `nifty` and value equal to that target. HTTPS turns on by itself once
   DNS has updated.
5. **Backup pinger (optional)**: at https://cron-job.org (free), create a job that opens
   `https://nifty.ujesh.in/healthz` every 10 minutes.
6. **Updating**: change anything on your PC, run step 1 again, commit and push in GitHub Desktop.
   Render redeploys by itself.

## Worth knowing

* NSE's website often blocks cloud servers, so on Render the price usually comes from Yahoo (the
  app switches by itself). Yahoo can be a few seconds to a minute behind; the screen shows the lag.
* The free CPU is small, so the page redraws every 2 seconds instead of 1. Forecasts still refresh
  every 5 seconds.
* A restart in the middle of a trading day re-loads today's bars from Yahoo, so the chart fills
  back in, but alerts from before the restart are gone.
* Free hours: 750 a month, which is enough for one app running all month.
