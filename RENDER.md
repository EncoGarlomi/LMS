Deploying the Telegram bot to Render

1) Create a new service on Render
- Service type: Background Worker (or Web Service if you prefer)
- Environment: Python 3.x
- Build Command: `pip install -r requirements.txt`
- Start Command: `python -m app.bot` (Procfile has `worker: python -m app.bot`)

2) Environment variables (set in Render dashboard)
- `BOT_TOKEN` (required)
- `ADMIN_ID` (optional)
- `MODEL_PATH` (optional) — if left empty, the repo will use `salesforce/ctrl` as upstream model; set to a path in the repo or a mounted location if you upload a trained checkpoint. Default in repo: `artifacts/ctrl-news`.
- `DATASET_NAME` (optional) — defaults to `shaurya03/tech-news-daily` in this repository.
- GPU: Render does not provide GPU by default. Training on Render is not supported without GPU. Use local GPU or a dedicated GPU host for training.

3) Files saved during runtime
- Render instances have ephemeral filesystem. If you want to persist fine-tuned checkpoints, push them to external storage (S3, GCS) or build an image that already contains the `artifacts/ctrl-news` folder.

4) Quick deploy steps
- Push this repo to GitHub
- Create a new Render Background Worker connected to the repo
- Set required environment variables in Render
- Deploy

Notes
- The bot will attempt to load the latest checkpoint under `MODEL_PATH` (checks for `checkpoint-*` directories or model files). If none found, it falls back to the upstream model name.
- For production, keep `BOT_TOKEN` secret and consider using a Redis-backed queue if you need horizontal scaling.
 - After changing environment variables in the Render dashboard, trigger a redeploy or use the `/reloadsettings` admin command to have the running bot re-read `.env` and reload the model without restarting (note: large model files won't appear unless present in the instance filesystem).

Suggested files included:
- `Procfile` — starts the bot as a background worker.
- `runtime.txt` — Python runtime hint.
- `.env.example` — example environment variables to copy into `.env` locally or use when configuring Render.
