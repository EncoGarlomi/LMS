# CTRL News Bot

Telegram bot that uses the CTRL language model with the `Yehor/news-headlines-ubercorpus` dataset.

## What it does
- Loads the CTRL model for text generation
- Fine-tunes on the news headlines dataset
- Answers user prompts through Telegram
- Keeps the bot token and admin ID in `.env`

## Setup
1. Create a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in:
   - `BOT_TOKEN`
   - `ADMIN_ID`
4. Run the bot:
   ```bash
   python -m app.bot
   ```

## Quick Commands
- Start the bot from Windows:
   ```bat
   run_bot.bat
   ```
- Train the model:
   ```bash
   python -m app.train
   ```
- Start the bot with the trained checkpoint in `MODEL_PATH`:
   ```bash
   python -m app.bot
   ```

If `MODEL_PATH` exists, the bot loads the fine-tuned checkpoint from there.

## Train locally
If you want to fine-tune the model on the dataset first:
```bash
python -m app.train
```

The trained checkpoint is saved under `artifacts/ctrl-news` by default.

## Bot usage
- Send any text to the bot to get a generated headline-style answer.
- `/start` shows the basic help.
- `/generate <prompt>` asks for a custom generation.
- `/status` is reserved for the admin account.

## Notes
- `MODEL_NAME` defaults to `salesforce/ctrl`.
- If `MODEL_PATH` exists, the bot loads the fine-tuned checkpoint from there.
- The dataset text column can be set with `TEXT_COLUMN`; if it is empty, the loader tries to detect one automatically.
