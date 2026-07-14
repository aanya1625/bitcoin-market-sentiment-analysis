# 📊 Bitcoin Market Sentiment Analysis: Trader Performance vs Fear & Greed Index

## Overview

This project analyzes the relationship between **Bitcoin market sentiment** and **Hyperliquid trader performance** by combining the **Bitcoin Fear & Greed Index** with historical trading data.

The objective is to understand how market sentiment influences trader profitability, trading activity, position direction, and overall trading behavior. The analysis uncovers actionable insights that can support data-driven trading strategies.

---

## Objectives

- Merge Bitcoin Fear & Greed Index with Hyperliquid trading data
- Analyze trader performance across different market sentiment regimes
- Compare profitability during Fear, Neutral, Greed, Extreme Fear, and Extreme Greed
- Examine long vs short trading behavior
- Generate business insights through exploratory data analysis (EDA)
- Produce a professional analytical report with visualizations

---

## Dataset

### 1. Bitcoin Fear & Greed Index

**Columns**

- Date
- Fear & Greed Score
- Classification
  - Extreme Fear
  - Fear
  - Neutral
  - Greed
  - Extreme Greed

### 2. Hyperliquid Historical Trading Data

Contains over **211,000+ trades** including:

- Account
- Coin
- Execution Price
- Position Size
- Side
- Direction
- Closed PnL
- Fee
- Timestamp
- Trade ID

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- JSON
- Jupyter Notebook

---


---

## Methodology

- Data Cleaning
- Timestamp Processing
- Dataset Merging
- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Data Visualization
- Business Insight Generation

---

## Key Analysis Performed

- Trader Win Rate by Market Sentiment
- Average Profit per Trade
- Total Realized PnL by Sentiment
- Trading Volume Analysis
- Long vs Short Position Analysis
- Daily Market Sentiment vs Profitability
- Top & Bottom Performing Traders
- Correlation Analysis

---

## Sample Visualizations

### Trader Win Rate by Market Sentiment

![Win Rate](charts/01_win_rate_by_sentiment.png)

---

### Average Trade Profitability

![Average Profit](charts/02_avg_pnl_by_sentiment.png)

---

### Total Trading Volume

![Volume](charts/06_volume_by_sentiment.png)

---

## Key Insights

- Extreme Greed produced the highest average profit per trade.
- Fear periods generated the largest overall realized profits.
- Neutral market conditions showed comparatively weaker profitability.
- Traders shifted from predominantly long positions during Fear to more short positions during Greed.
- Market sentiment categories provided stronger behavioral insights than the raw Fear & Greed score alone.

---

## Skills Demonstrated

- Data Cleaning
- Data Preprocessing
- Data Merging
- Exploratory Data Analysis
- Statistical Analysis
- Financial Data Analytics
- Data Visualization
- Business Intelligence
- Python Programming

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/bitcoin-market-sentiment-analysis.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the analysis

```bash
python trader_sentiment_analysis.py
```

---

## Output

The project generates:

- Analytical Report (PDF)
- Charts
- Daily Aggregated Dataset
- JSON Summary of Results

---

## Future Improvements

- Interactive Streamlit Dashboard
- Predictive Machine Learning Models
- Portfolio Optimization
- Risk Analysis
- Time-Series Forecasting
- Real-time Market Sentiment Integration

---

## Author

**Aanya Jain**

B.Tech Electronics & Communication Engineering  
Aspiring Data Analyst | AI & Data Analytics | Python | SQL | Machine Learning

---

⭐ If you found this project interesting, consider giving it a star!
