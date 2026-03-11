"""AI Research Agent — Financial research with web search, stock data, and analysis tools."""

import streamlit as st
from agent.core import run_agent, MODELS
from agent.tools import TOOL_DEFINITIONS

st.set_page_config(page_title="AI Research Agent", page_icon="🤖", layout="wide")

EXAMPLE_QUERIES = [
    "Compare Tencent (0700.HK) and Alibaba (9988.HK) — which is a better investment right now?",
    "What are the latest developments in the Hong Kong stock market?",
    "Analyze NVIDIA's financial health and recent stock performance.",
    "What is the current state of the AI chip market and key players?",
]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "HuggingFace Token",
        type="password",
        help="Free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)",
    )
    model = st.selectbox("LLM Model", MODELS)

    st.divider()
    st.subheader("🛠️ Available Tools")
    for tool in TOOL_DEFINITIONS:
        with st.expander(f"**{tool['name']}**"):
            st.write(tool["description"])
            for k, v in tool["parameters"].items():
                st.code(f"{k}: {v}", language=None)

# ── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🤖 AI Research Agent")
st.caption(
    "A ReAct agent that reasons step-by-step, calls tools (web search, stock data, calculator), "
    "and synthesizes a final answer. Powered by HuggingFace Inference API."
)

# ── How it works ─────────────────────────────────────────────────────────────
with st.expander("How it works — ReAct Framework"):
    st.markdown(
        """
**ReAct** (Reason + Act) is an agent framework where the LLM alternates between:

1. **Thought** — Reason about what information is needed
2. **Action** — Call a tool (web search, stock API, calculator)
3. **Observation** — Receive the tool's output
4. **Repeat** until enough information is gathered
5. **Final Answer** — Synthesize all observations into a comprehensive response

This agent has access to:
- **web_search** — DuckDuckGo search for news and information
- **get_stock_price** — Real-time stock prices via yfinance
- **get_stock_financials** — Key financial metrics (P/E, revenue, margins)
- **calculate** — Math expressions for ratios and comparisons
"""
    )

st.divider()

# ── Example queries ──────────────────────────────────────────────────────────
st.subheader("Try an example")
cols = st.columns(2)
for i, example in enumerate(EXAMPLE_QUERIES):
    if cols[i % 2].button(example, key=f"ex_{i}", use_container_width=True):
        st.session_state["prefill_query"] = example

# ── Query input ──────────────────────────────────────────────────────────────
query = st.text_area(
    "Ask a research question",
    value=st.session_state.pop("prefill_query", ""),
    height=80,
    placeholder="e.g. Compare Tesla and BYD's financial performance...",
)

run_btn = st.button("🚀 Run Agent", type="primary", disabled=not query)

# ── Run agent ────────────────────────────────────────────────────────────────
if run_btn and query:
    if not api_key:
        st.error("Please enter your HuggingFace token in the sidebar.")
        st.stop()

    st.divider()
    st.subheader("Agent Reasoning Trace")

    steps_container = st.container()
    status = st.status("Agent is thinking...", expanded=True)

    with status:
        steps = run_agent(api_key, model, query)

    # Display steps
    final_answer = None
    with steps_container:
        for s in steps:
            if s["type"] == "thought" and s["content"]:
                st.markdown(f"💭 **Thought:** {s['content']}")
            elif s["type"] == "action":
                st.markdown(f"🔧 **Action:** `{s['tool']}` → `{s['input']}`")
            elif s["type"] == "observation":
                with st.expander("📋 Observation (tool output)"):
                    st.text(s["content"][:2000])
            elif s["type"] == "final_answer":
                final_answer = s["content"]
            elif s["type"] == "error":
                st.warning(f"⚠️ {s['content']}")

    status.update(label="Agent finished!", state="complete", expanded=False)

    if final_answer:
        st.divider()
        st.subheader("📊 Final Answer")
        st.markdown(final_answer)

        # Save to history
        st.session_state.history.append({
            "query": query,
            "answer": final_answer,
            "steps": len([s for s in steps if s["type"] == "action"]),
        })

# ── History ──────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader("📜 Query History")
    for i, h in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Q: {h['query'][:80]}... ({h['steps']} tool calls)"):
            st.markdown(h["answer"])
