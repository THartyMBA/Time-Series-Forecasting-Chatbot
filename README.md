# Time-Series-Forecasting-Chatbot
🔮 Time-Series Forecasting + Data-Aware Chatbot
Upload → Forecast → Chat – all in a single Streamlit page.

This app combines the Time-Series Forecaster with an LLM chatbot (powered by a free OpenRouter model) that actually knows about the data and the forecast you just created.

🏃‍♂️ What you can do in 30 seconds
Upload a CSV that has

one date column

one numeric value column

Choose horizon & frequency → click Run forecast.

Trains Prophet (auto-ARIMA fallback)

Shows an interactive Plotly chart & confidence interval

Ask questions like

“What’s the predicted value next quarter?”

“Why does the forecast dip in 2023?”
The chatbot receives lightweight snapshots of your dataframe and the forecast, so it can answer context-aware questions in real time.

Download the forecast CSV whenever you want.

Proof-of-concept – no auth, no PII masking, no long-term persistence.
Need enterprise-grade forecasting or RAG chat? → drtomharty.com/bio

✨ Features

Module	Highlights
Forecaster	Prophet → auto-ARIMA fallback; daily/weekly/monthly/hourly; cached I/O.
LLM Chat	Uses any free OpenRouter model (default mistralai/mistral-7b-instruct). Full conversational memory per browser tab.
Context injection	Sends the top of the raw dataframe & tail of the forecast as hidden system messages so the bot can reference your data without hitting token limits.
Downloadables	Forecast CSV with one click.
🔑 Secrets
Add your OpenRouter API key.


Environment	How
Streamlit Cloud	App Dashboard → ⋯ ➜ Edit secrets
OPENROUTER_API_KEY = "sk-or-xxxxxxxx"
Local dev	~/.streamlit/secrets.toml or export OPENROUTER_API_KEY=sk-or-xxxxxxxx
🚀 Quick start (local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/Time-Series-Forecasting-Chatbot.git
cd Time-Series_Forecasting-Chatbot
python -m venv venv && source venv/bin/activate     # Win: venv\Scripts\activate
pip install -r requirements.txt
streamlit run unified_app.py
Browse to http://localhost:8501 and play.

☁️ Deploy on Streamlit Cloud (free)
Push the repo to GitHub.

Go to streamlit.io/cloud ➜ New app → select repo/branch.

Add OPENROUTER_API_KEY in Secrets.

Deploy and share your public URL.

(Free tier CPU is plenty – Prophet + ARIMA train in ≤2 s on demo datasets.)

🗂️ Repo layout
java
Copy
Edit
unified_app.py      ← the entire app (forecast + chat)
requirements.txt    ← core deps
README.md           ← you’re here
🛠️ Requirements
shell
Copy
Edit
streamlit>=1.32
pandas
numpy
plotly
prophet
pmdarima
requests
(prophet wheel builds automatically on Streamlit Cloud; falls back to auto-ARIMA if not.)

📜 License
CC0 – public-domain dedication. Credit is welcome but not required.

🙏 Acknowledgements
Streamlit – zero-friction data apps

Prophet & pmdarima – forecasting

OpenRouter – free LLM gateway

Plotly – interactive charts

Forecast, converse, iterate – all in one tab. Enjoy! 🎉
