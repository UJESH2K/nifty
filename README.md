# NiftySim

Watch your own model predict NIFTY 50 live, from 1 minute up to 3 hours ahead, drawn over the real
market. It keeps score of whether it's actually right. Everything is free.

> Research and education only. This is not investment advice and not a trading system. If you ever
> share its signals publicly in India, SEBI's Research Analyst / Investment Adviser rules apply.

---

## Tonight (10 minutes)

1. **Unzip** so you have one folder, e.g. `C:\Users\you\Downloads\niftysim\`, with `config.yaml`,
   `README.md` and `start_niftysim.bat` directly inside it. If you get `niftysim\niftysim\…`, use
   the inner folder. Every command below runs from the folder that contains `config.yaml`.
2. **All available history is downloaded automatically** (1-minute data from 2015 to the latest
   update, about 11 years; free minute-level NIFTY data doesn't go further back) from the free Kaggle dataset *NSE – Nifty 50
   Index Minute data* (via `kagglehub`). If that download fails, setup carries on with ~29 days
   from Yahoo, and you can download the CSV by hand, save it as `nifty_50_minute.csv` in the
   folder, and run setup again.
3. **Double-click `setup_tonight.bat`.** It installs the packages, downloads real NIFTY data, collects
   headlines and trains the models.
   *(Command line: `pip install -r requirements.txt`, then `python scripts/setup_real_data.py`.)*

Want to see the screens first? `try_demo.bat` runs everything on a simulated market.
Don't mix the two: `setup_tonight.bat` moves demo files into `data/backup_*` automatically.

## Tomorrow morning

1. Double-click **`start_niftysim.bat`** any time before 09:15. The browser opens at
   http://127.0.0.1:8050.
2. Leave the source on **NSE + Yahoo (no account)** and press **Go live**. It reads NSE's own website
   and switches to Yahoo by itself if NSE blocks it. **No demat account is needed.**
3. Leave the tab open. The ring counts down to the open. At 09:15 IST it loads today's bars and
   starts predicting on its own.
4. Pick a horizon: **1m, 5m, 10m, 15m, 30m, 1h, 2h, 3h**, or **Custom** (any minute up to 3 h).
   The lavender line is the predicted path, the shaded band is where price should land 80% of the
   time, and the dotted line shows what the model said that long ago, so you can see whether it
   worked.
5. "How it's doing today" fills in once the first predictions come due. It compares the model's
   average miss with a plain "no change" guess, which is the honest bar to clear.

Every tick, 1-second bar and 1-minute bar is saved, so running `setup_tonight.bat` again (or Lab,
then Retrain now) teaches the model from today.

## API keys: what you actually need

**Nothing, and no demat account.** Prices come from NSE's public website (with Yahoo as backup),
news from GDELT and RSS. Angel One and Upstox are only for people who already have a demat account.
Add these only if you want them:

| What it gives you | Where to get it | `.env` line(s) | Notes |
|---|---|---|---|
| Years of 1-min history (best for training) | [Kaggle: NSE – Nifty 50 Index Minute data](https://www.kaggle.com/datasets/debashis74017/nifty-50-minute-data) | none | Free Kaggle login to download |
| **True tick-by-tick live prices (recommended)** | [Angel One SmartAPI](https://smartapi.angelone.in/), then *Create an app* | `ANGEL_API_KEY` | Free. Needs an Angel One account |
| | Your Angel One client ID | `ANGEL_CLIENT_CODE` | |
| | Your Angel One **MPIN** | `ANGEL_PIN` | |
| | [Enable TOTP](https://smartapi.angelbroking.com/enable-totp) (client ID + MPIN + OTP); copy the **text code under the QR** | `ANGEL_TOTP_SECRET` | One-time setup; after that it logs in by itself every day |
| Alternative live prices | [Upstox developer apps](https://account.upstox.com/developer/apps) · [docs](https://upstox.com/developer/api-documentation/) | `UPSTOX_ACCESS_TOKEN` | Free, but the token expires daily, so you log in each morning |
| Extra news | [Finnhub](https://finnhub.io/register) | `FINNHUB_API_KEY` | Free tier |
| Extra Indian news | [Marketaux](https://www.marketaux.com/register) | `MARKETAUX_API_KEY` | Free tier, ~100 calls/day |

Copy `.env.example` to `.env` (same folder as `config.yaml`) and fill in only the lines you use.
Never share `.env`.

**Enough to make it work:** nothing. For better 1–3 h predictions, add the free Kaggle history (only
needs a Kaggle login, not a demat account). Yahoo's index quotes can lag the exchange a little; the app shows the lag if it exceeds
20 s. The Yahoo, Angel One and Upstox connectors couldn't be tested from the build environment. If
one misbehaves, set `NIFTYSIM_DEBUG=1` in `.env` and send the console output.

## Live layer: every 5 seconds, plus critical-moment alerts

* Every **5 seconds** every forecast is recomputed from the candle that's still forming, glued to the
  latest price (set `alerts.refresh_seconds` in `config.yaml`).
* A **critical moment** is any of: a 5-minute move about 3.5× faster than usual, the expected range
  suddenly widening (volatility breakout), a high-importance headline, or the model projecting a big
  move (2× normal volatility or more). Then:
  1. a **"Be prepared"** banner shows the expected points, the likely range and, for big directional
     calls, your **one-third target** (projects +300, aim for +100)
  2. a **pop-up notification with a beep** (press *Turn on pop-up alerts* once in the browser)
  3. the model is **re-tuned on the spot** on the last 60 days plus everything seen today, and used
     50/50 with the main model for the rest of the day; the range is widened slightly to stay safe
* At most one alert per 15 minutes. Every threshold is in `config.yaml` under `alerts:`.

## Running it unattended and on your domain

`start_autopilot.bat` (or `python scripts/run_dashboard.py --autopilot auto`) goes live every
trading day, saves everything, scores the day, and retrains overnight with nobody touching it. Lab
shows the day-by-day "Is it getting better?" chart. To put it on `nifty.ujesh.in`, follow
`deploy/DEPLOY.md`. A password is optional: set `NIFTYSIM_PASSWORD` in `.env` only if you want one.

## The three screens

* **Live:** the Go live orb and countdown, then the live chart, prediction panel, today's
  scorecard, every tick, and headlines.
* **Blind replay:** pick a past session. A fresh model is trained **only on days before it**, then
  the day plays back (real time, 10×, 60×, 300×, or Max) with its headlines. You watch the
  predictions form and get checked, ending with a table for every horizon. This is the honest test of
  "would it have worked?".
* **Lab:** scrub through history, switch models, retrain. A banner tells you when you're looking at
  days the model trained on, which look better than reality.

## How the models stay honest

* **Two models side by side:** *price only* and *with news* (headlines scored for direction and
  importance). Pick **Compare** to see both.
* **Direction is earned, not assumed.** While training, each model is tested on days it hasn't seen.
  If its directional calls at some horizon didn't help, they're scaled toward "no change", so it
  gives only a range there. On a month of data that usually happens beyond ~45 minutes. With years
  of Kaggle data it may keep more.
* **Ranges self-correct during the day.** If price lands outside the 80% band too often, the band
  widens, and if it's too wide it tightens. Weights are retrained only overnight.
* **Expect this:** at 1–20 minutes, price-only is roughly a coin flip. The news model can do a
  little better after headlines. Ranges (how far it might move) are much more reliable than
  direction.

## Choosing a model

`ridge` (default, fast and stable) · `lgbm` (non-linear, slower to train) · `naive` (no direction,
just volatility, the baseline) · `mlp` (experimental). Switch in Lab or with
`start_niftysim.bat --model lgbm`. To add your own, subclass `ForecastModel` in
`niftysim/models/__init__.py` and register it.

## Files

```
start_niftysim.bat / setup_tonight.bat / try_demo.bat   double-click launchers (Windows)
config.yaml            horizons, holidays, feeds, news, rules. Edit here
data/                  your prices, ticks and headlines (events.csv = RBI/Budget dates you add)
scripts/               setup_real_data, run_dashboard, train_models, download_history, fetch_news, record_ticks
niftysim/              engine, models, feeds (yahoo/angel/upstox/replay/demo), news, market clock
app/                   the dashboard
```

NSE holidays for 2026 are in `config.yaml`. The clock skips them (for example Tue 24 Nov). Add
2027's list when NSE publishes it.
