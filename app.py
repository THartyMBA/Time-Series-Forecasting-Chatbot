# unified_app.py
"""
🔮 Data-Aware Chatbot + Time-Series Forecasting
---------------------------------------------------------------------------

A single Streamlit app that lets visitors…

1. **Upload** any CSV with one date column + numeric column(s).  
2. **Run** an on-the-fly forecast (Prophet → auto-ARIMA fallback).  
3. **Chat** with a memory-enabled OpenRouter model about the data *and* the
   forecast results.

Everything (chat history, raw data, forecast) is kept in `st.session_state`,
so the bot can reference prior context throughout the session.

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
REQUIRED SECRET OR ENV VAR → `OPENROUTER_API_KEY`
––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
"""

import os
import io
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from pmdarima import auto_arima
from datetime import datetime

# ╭─────────────────────────────────────────────────────────╮
#                       CONFIG / KEYS                       #
# ╰─────────────────────────────────────────────────────────╯
OPENROUTER_API_KEY = (
    st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
)
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"  # free tier

# ╭─────────────────────────────────────────────────────────╮
#                          CHAT API                         #
# ╰─────────────────────────────────────────────────────────╯
def openrouter_chat(messages, model=DEFAULT_MODEL, temperature=0.7):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in env or st.secrets.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-portfolio.example",  # change if desired
        "X-Title": "DataChatForecastDemo",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ╭─────────────────────────────────────────────────────────╮
#                    FORECASTING UTILITIES                  #
# ╰─────────────────────────────────────────────────────────╯
def run_prophet(df, date_col, val_col, periods, freq):
    data = df[[date_col, val_col]].dropna().copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.rename(columns={date_col: "ds", val_col: "y"}).dropna()
    m = Prophet()
    m.fit(data)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fc, m


def run_auto_arima(df, date_col, val_col, periods, freq):
    ts = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col], errors="coerce")).dropna()
    model = auto_arima(ts, seasonal=False, suppress_warnings=True, stepwise=True, error_action="ignore")
    preds, conf = model.predict(periods, return_conf_int=True)
    idx = pd.date_range(ts.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    fc = pd.DataFrame(
        {"ds": idx, "yhat": preds, "yhat_lower": conf[:, 0], "yhat_upper": conf[:, 1]}
    )
    return fc, model


def plot_forecast(hist_df, date_col, val_col, fc_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_df[date_col], y=hist_df[val_col], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"], name="Forecast", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=fc_df["ds"],
            y=fc_df["yhat_upper"],
            name="Upper CI",
            mode="lines",
            line=dict(dash="dash"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_df["ds"],
            y=fc_df["yhat_lower"],
            name="Lower CI",
            mode="lines",
            line=dict(dash="dash"),
            fill="tonexty",
            showlegend=False,
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=450)
    return fig


# ╭─────────────────────────────────────────────────────────╮
#                 SESSION INITIALIZATION                    #
# ╰─────────────────────────────────────────────────────────╯
def init_session():
    if "chat" not in st.session_state:
        st.session_state.chat = [
            {
                "role": "system",
                "content": (
                    "You are DataChat, a helpful data-science assistant. "
                    "If the user has uploaded data or a forecast, you can reference it via "
                    "`df` or `forecast_df` to answer questions."
                ),
            }
        ]
    # placeholders for data & forecast
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("forecast_df", None)
    st.session_state.setdefault("forecast_plot", None)
    st.session_state.setdefault("model_info", "")


# ╭─────────────────────────────────────────────────────────╮
#                        STREAMLIT UI                       #
# ╰─────────────────────────────────────────────────────────╯
st.set_page_config(page_title="🔮 Data-Aware Chatbot", layout="wide")
init_session()

# Sidebar – data upload & forecasting
with st.sidebar:
    st.header("📂 Upload / Forecast")
    file = st.file_uploader("CSV with a date column + numeric column", type="csv")

    if file:
        st.session_state.df = pd.read_csv(file)
        st.success("File loaded! Choose columns below.")

        date_col = st.selectbox("Date column", st.session_state.df.columns)
        num_col = st.selectbox(
            "Value column (to forecast)", st.session_state.df.select_dtypes(include="number").columns
        )
        horizon = st.number_input("Periods to forecast", 1, 365 * 3, 30)
        freq = st.selectbox("Frequency", options=["D", "W", "M", "H"], index=0)

        if st.button("🚀 Run forecast"):
            try:
                fc, mdl = run_prophet(st.session_state.df, date_col, num_col, horizon, freq)
                engine = "Prophet"
            except Exception:
                fc, mdl = run_auto_arima(st.session_state.df, date_col, num_col, horizon, freq)
                engine = "auto-ARIMA"

            st.session_state.forecast_df = fc
            st.session_state.forecast_plot = plot_forecast(st.session_state.df, date_col, num_col, fc)
            st.session_state.model_info = (
                f"Forecast created on **{datetime.now():%Y-%m-%d %H:%M:%S}** "
                f"with **{engine}** ({horizon} {freq}-steps)."
            )
            st.success("Forecast complete! Discuss it in the chat →")

# Main area – chat plus optional visuals
st.title("🔮 Data-Aware Chatbot")
st.info(
    "🔔 **Demo Notice**  \n"
    "This application is a streamlined proof-of-concept, **not** an "
    "enterprise-grade product.  \n\n"
    "Need production-level performance, security or custom features? "
    "[Get in touch](mailto:you@example.com) and let’s build a tailored solution.",
    icon="💡",
)

# Show plot if available
if st.session_state.forecast_plot is not None:
    st.subheader("Forecast")
    st.plotly_chart(st.session_state.forecast_plot, use_container_width=True)
    st.caption(st.session_state.model_info)

# Display chat history
for msg in st.session_state.chat[1:]:  # skip system prompt
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
user_msg = st.chat_input("Ask me about your data or anything…")
if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    st.chat_message("user").markdown(user_msg)

    # 💡– Attach small context snippets (head of df & head of forecast) for grounding
    if st.session_state.df is not None:
        df_head = st.session_state.df.head().to_markdown(index=False)
        st.session_state.chat.append(
            {
                "role": "system",
                "content": f"Here is the first few rows of the uploaded dataframe:\n\n{df_head}",
            }
        )
    if st.session_state.forecast_df is not None:
        fc_head = st.session_state.forecast_df.tail(5).to_markdown(index=False)
        st.session_state.chat.append(
            {
                "role": "system",
                "content": f"Here are the last 5 rows of the forecast dataframe:\n\n{fc_head}",
            }
        )

    # Call OpenRouter
    with st.spinner("Thinking…"):
        try:
            reply = openrouter_chat(st.session_state.chat)
        except Exception as e:
            st.error(f"OpenRouter error: {e}")
            st.stop()

    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").markdown(reply)
