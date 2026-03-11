"""Tools that the agent can call — web search, stock data, calculator."""

import json
from duckduckgo_search import DDGS
import yfinance as yf
from datetime import datetime, timedelta


TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": "Search the web for recent information. Use this for news, company info, market trends, or any factual question.",
        "parameters": {
            "query": "The search query string",
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get current and recent stock price data for a given ticker symbol (e.g. 0700.HK for Tencent, AAPL for Apple).",
        "parameters": {
            "ticker": "Stock ticker symbol (e.g. AAPL, 0700.HK, 9988.HK)",
        },
    },
    {
        "name": "get_stock_financials",
        "description": "Get key financial metrics for a stock: market cap, P/E ratio, revenue, profit margins.",
        "parameters": {
            "ticker": "Stock ticker symbol",
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use for any calculations like ratios, percentages, growth rates.",
        "parameters": {
            "expression": "A Python math expression to evaluate (e.g. '100 * 1.05 ** 10')",
        },
    },
]


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}")
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {e}"


def get_stock_price(ticker: str) -> str:
    """Get recent stock price data."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        if hist.empty:
            return f"No data found for ticker '{ticker}'."

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

        info = {
            "ticker": ticker,
            "latest_close": round(float(latest["Close"]), 2),
            "previous_close": round(float(prev["Close"]), 2),
            "daily_change_pct": round(change, 2),
            "volume": int(latest["Volume"]),
            "30d_high": round(float(hist["High"].max()), 2),
            "30d_low": round(float(hist["Low"].min()), 2),
            "date": str(hist.index[-1].date()),
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error fetching stock data: {e}"


def get_stock_financials(ticker: str) -> str:
    """Get key financial metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        metrics = {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "revenue": info.get("totalRevenue", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        }
        return json.dumps(metrics, indent=2)
    except Exception as e:
        return f"Error fetching financials: {e}"


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/.() %e")
        clean = expression.replace("**", "^").replace("^", "**")
        result = eval(clean, {"__builtins__": {}}, {"abs": abs, "round": round, "min": min, "max": max})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "get_stock_price": get_stock_price,
    "get_stock_financials": get_stock_financials,
    "calculate": calculate,
}
