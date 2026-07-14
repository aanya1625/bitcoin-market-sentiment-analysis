"""
Trader Performance vs. Bitcoin Market Sentiment
=================================================
Analyzes the relationship between Hyperliquid trader performance and the
Bitcoin Fear & Greed Index.

Inputs (place in the same folder as this script, or edit the paths below):
    - fear_greed_index.csv   : columns -> timestamp, value, classification, date
    - historical_data.csv    : columns -> Account, Coin, Execution Price, Size Tokens,
                                Size USD, Side, Timestamp IST, Start Position, Direction,
                                Closed PnL, Transaction Hash, Order ID, Crossed, Fee,
                                Trade ID, Timestamp

Outputs:
    - daily_series.csv       : daily aggregated sentiment / volume / PnL series
    - results.json           : all computed summary metrics
    - charts/*.png           : 7 charts used in the accompanying report

Usage:
    python trader_sentiment_analysis.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
FEAR_GREED_PATH = "fear_greed_index.csv"
HISTORICAL_PATH = "historical_data.csv"
OUTPUT_DIR = "."
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

ORDER_5 = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
COLORS_5 = ["#8B0000", "#E76F51", "#BDBDBD", "#8AB17D", "#1B6B4A"]


def bucket_sentiment(classification: str) -> str:
    """Collapse the 5 official Fear & Greed classes into 3 broad buckets."""
    if classification in ("Extreme Fear", "Fear"):
        return "Fear"
    if classification in ("Extreme Greed", "Greed"):
        return "Greed"
    return "Neutral"


def load_and_merge(fg_path: str, hist_path: str) -> pd.DataFrame:
    """Load both datasets and merge trades onto the sentiment classification
    for the calendar date each trade executed on."""
    fg = pd.read_csv(fg_path)
    fg["date"] = pd.to_datetime(fg["date"])
    fg = fg[["date", "classification", "value"]].rename(columns={"value": "fg_value"})

    hist = pd.read_csv(hist_path)
    hist["Timestamp IST"] = pd.to_datetime(hist["Timestamp IST"], format="%d-%m-%Y %H:%M")
    hist["date"] = hist["Timestamp IST"].dt.normalize()

    df = hist.merge(fg, on="date", how="inner")
    df["sentiment_bucket"] = df["classification"].apply(bucket_sentiment)
    df["classification"] = pd.Categorical(df["classification"], categories=ORDER_5, ordered=True)
    df["sentiment_bucket"] = pd.Categorical(
        df["sentiment_bucket"], categories=["Fear", "Neutral", "Greed"], ordered=True
    )
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute all summary statistics used in the report."""
    results = {}
    closed = df[df["Closed PnL"] != 0].copy()
    closed["win"] = closed["Closed PnL"] > 0

    results["summary"] = {
        "total_trades": int(len(df)),
        "unique_accounts": int(df["Account"].nunique()),
        "unique_coins": int(df["Coin"].nunique()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "total_volume_usd": float(df["Size USD"].sum()),
        "total_realized_pnl": float(df["Closed PnL"].sum()),
        "total_fees": float(df["Fee"].sum()),
    }

    # Performance by 5-class sentiment
    by_sent = df.groupby("classification", observed=True).agg(
        trade_count=("Closed PnL", "size"),
        total_volume_usd=("Size USD", "sum"),
        avg_trade_size_usd=("Size USD", "mean"),
        total_fees=("Fee", "sum"),
    )
    closed_by_sent = closed.groupby("classification", observed=True).agg(
        closed_trades=("Closed PnL", "size"),
        total_realized_pnl=("Closed PnL", "sum"),
        avg_pnl_per_trade=("Closed PnL", "mean"),
        median_pnl_per_trade=("Closed PnL", "median"),
        win_rate=("win", "mean"),
    )
    sent_table = by_sent.join(closed_by_sent)
    sent_table["win_rate"] *= 100
    results["by_sentiment"] = sent_table.reset_index().to_dict("records")

    # Long/short new-position bias
    opens = df[df["Direction"].isin(["Open Long", "Open Short"])].copy()
    opens["side_group"] = opens["Direction"].map({"Open Long": "Long", "Open Short": "Short"})
    ls = opens.groupby(["classification", "side_group"], observed=True).size().unstack(fill_value=0)
    ls["long_pct"] = ls["Long"] / (ls["Long"] + ls["Short"]) * 100
    results["long_short_bias"] = ls.reset_index().to_dict("records")

    # Long/short profitability by sentiment
    closes_dir = closed[closed["Direction"].isin(["Close Long", "Close Short"])].copy()
    closes_dir["side_group"] = closes_dir["Direction"].map({"Close Long": "Long", "Close Short": "Short"})
    side_perf = closes_dir.groupby(["classification", "side_group"], observed=True).agg(
        trades=("Closed PnL", "size"),
        avg_pnl=("Closed PnL", "mean"),
        win_rate=("win", "mean"),
        total_pnl=("Closed PnL", "sum"),
    )
    side_perf["win_rate"] *= 100
    results["side_performance"] = side_perf.reset_index().to_dict("records")

    # Daily time series + correlations
    daily = df.groupby("date").agg(
        fg_value=("fg_value", "first"),
        classification=("classification", "first"),
        volume=("Size USD", "sum"),
    )
    daily_pnl = closed.groupby("date")["Closed PnL"].sum().rename("daily_pnl")
    daily = daily.join(daily_pnl).fillna({"daily_pnl": 0}).reset_index()
    daily.to_csv(os.path.join(OUTPUT_DIR, "daily_series.csv"), index=False)

    results["correlations"] = {
        "fg_value_vs_daily_pnl": float(daily["fg_value"].corr(daily["daily_pnl"])),
        "fg_value_vs_daily_volume": float(daily["fg_value"].corr(daily["volume"])),
    }

    # Account concentration
    acct_total = closed.groupby("Account")["Closed PnL"].sum().sort_values(ascending=False)
    results["top_accounts"] = acct_total.head(5).to_dict()
    results["bottom_accounts"] = acct_total.tail(5).to_dict()

    return results, daily


def make_charts(df: pd.DataFrame, daily: pd.DataFrame, out_dir: str) -> None:
    """Generate the 7 charts used in the accompanying report."""
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    closed = df[df["Closed PnL"] != 0].copy()
    closed["win"] = closed["Closed PnL"] > 0

    # 1. Win rate by sentiment
    win_rate = closed.groupby("classification", observed=True)["win"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(win_rate.index, win_rate.values, color=COLORS_5)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Trader Win Rate by Market Sentiment", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, win_rate.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_win_rate_by_sentiment.png", dpi=150)
    plt.close()

    # 2. Avg PnL per trade by sentiment
    avg_pnl = closed.groupby("classification", observed=True)["Closed PnL"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(avg_pnl.index, avg_pnl.values, color=COLORS_5)
    ax.set_ylabel("Average Closed PnL per Trade (USD)")
    ax.set_title("Average Trade Profitability by Market Sentiment", fontsize=13, fontweight="bold")
    for b, v in zip(bars, avg_pnl.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"${v:.0f}", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/02_avg_pnl_by_sentiment.png", dpi=150)
    plt.close()

    # 3. Total realized PnL by sentiment
    total_pnl = closed.groupby("classification", observed=True)["Closed PnL"].sum() / 1e6
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(total_pnl.index, total_pnl.values, color=COLORS_5)
    ax.set_ylabel("Total Realized PnL (Million USD)")
    ax.set_title("Total Realized Profit by Market Sentiment", fontsize=13, fontweight="bold")
    for b, v in zip(bars, total_pnl.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"${v:.2f}M", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/03_total_pnl_by_sentiment.png", dpi=150)
    plt.close()

    # 4. Long vs short bias by sentiment
    opens = df[df["Direction"].isin(["Open Long", "Open Short"])].copy()
    opens["side_group"] = opens["Direction"].map({"Open Long": "Long", "Open Short": "Short"})
    ls = opens.groupby(["classification", "side_group"], observed=True).size().unstack(fill_value=0)
    ls_pct = ls.div(ls.sum(axis=1), axis=0).reindex(ORDER_5) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ls_pct.index, ls_pct["Long"], label="Long", color="#2A9D8F")
    ax.bar(ls_pct.index, ls_pct["Short"], bottom=ls_pct["Long"], label="Short", color="#E76F51")
    ax.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Share of New Positions Opened (%)")
    ax.set_title("Long vs Short Positioning by Market Sentiment", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/04_long_short_bias.png", dpi=150)
    plt.close()

    # 5. Sentiment vs daily PnL time series
    daily_sorted = daily.sort_values("date")
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(daily_sorted["date"], daily_sorted["fg_value"], color="#264653", linewidth=1.2)
    ax1.set_ylabel("Fear & Greed Index Value", color="#264653")
    ax1.tick_params(axis="y", labelcolor="#264653")
    ax1.axhline(25, color="#8B0000", linewidth=0.6, linestyle=":", alpha=0.6)
    ax1.axhline(75, color="#1B6B4A", linewidth=0.6, linestyle=":", alpha=0.6)
    ax2 = ax1.twinx()
    ax2.bar(daily_sorted["date"], daily_sorted["daily_pnl"] / 1000, color="#E9C46A", alpha=0.5, width=1.0)
    ax2.set_ylabel("Daily Realized PnL (thousand USD)", color="#B8860B")
    ax2.tick_params(axis="y", labelcolor="#B8860B")
    ax1.set_title("Market Sentiment vs Daily Trader PnL Over Time", fontsize=13, fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig(f"{out_dir}/05_sentiment_pnl_timeseries.png", dpi=150)
    plt.close()

    # 6. Volume by sentiment
    vol = df.groupby("classification", observed=True)["Size USD"].sum() / 1e6
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(vol.index, vol.values, color=COLORS_5)
    ax.set_ylabel("Total Trading Volume (Million USD)")
    ax.set_title("Trading Volume by Market Sentiment", fontsize=13, fontweight="bold")
    for b, v in zip(bars, vol.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"${v:.0f}M", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/06_volume_by_sentiment.png", dpi=150)
    plt.close()

    # 7. Long vs short avg PnL by sentiment
    closes_dir = closed[closed["Direction"].isin(["Close Long", "Close Short"])].copy()
    closes_dir["side_group"] = closes_dir["Direction"].map({"Close Long": "Long", "Close Short": "Short"})
    pivot = closes_dir.groupby(["classification", "side_group"], observed=True)["Closed PnL"].mean().unstack().reindex(ORDER_5)
    x = np.arange(len(ORDER_5))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, pivot["Long"], width, label="Long", color="#2A9D8F")
    ax.bar(x + width / 2, pivot["Short"], width, label="Short", color="#E76F51")
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER_5, rotation=15)
    ax.set_ylabel("Average Closed PnL per Trade (USD)")
    ax.set_title("Long vs Short Average Profitability by Sentiment", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/07_long_short_avg_pnl.png", dpi=150)
    plt.close()


def main():
    print("Loading and merging datasets...")
    df = load_and_merge(FEAR_GREED_PATH, HISTORICAL_PATH)
    print(f"Merged {len(df):,} trades across {df['Account'].nunique()} accounts.")

    print("Computing metrics...")
    results, daily = compute_metrics(df)
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Generating charts...")
    make_charts(df, daily, CHART_DIR)

    print("Done. See results.json, daily_series.csv, and charts/ for output.")


if __name__ == "__main__":
    main()
