"""ReAct agent — Reason + Act loop with tool calling via HuggingFace Inference API."""

import re
from huggingface_hub import InferenceClient
from agent.tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

SYSTEM_PROMPT = """You are a financial research agent with access to tools. Use the ReAct framework: Thought → Action → Observation → ... → Final Answer.

Available tools:
{tools}

IMPORTANT: You must respond in EXACTLY this format for each step:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <input for the tool>

When you have enough information to answer, respond with:
Thought: I now have enough information to answer.
Final Answer: <your comprehensive answer>

Rules:
- Always start with a Thought before taking an action
- Use tools to gather real data — do not make up numbers
- You can call multiple tools across steps
- Cite your sources in the final answer
- Be concise but thorough
"""

MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

MAX_STEPS = 8


def format_tools_description():
    parts = []
    for t in TOOL_DEFINITIONS:
        params = ", ".join(f"{k}: {v}" for k, v in t["parameters"].items())
        parts.append(f"- **{t['name']}**({params}): {t['description']}")
    return "\n".join(parts)


def parse_action(text):
    """Parse Action and Action Input from the LLM response."""
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)

    if action_match and input_match:
        return action_match.group(1).strip(), input_match.group(1).strip()
    return None, None


def run_agent(api_key, model, user_query, on_step=None):
    """Run the ReAct agent loop.

    on_step: optional callback(step_num, step_type, content) for streaming updates.
    """
    client = InferenceClient(api_key=api_key)
    tools_desc = format_tools_description()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_desc)},
        {"role": "user", "content": user_query},
    ]

    steps = []

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        # Check for Final Answer
        if "Final Answer:" in reply:
            final = reply.split("Final Answer:")[-1].strip()
            thought = ""
            if "Thought:" in reply:
                thought = reply.split("Thought:")[1].split("Final Answer:")[0].strip()
            steps.append({"type": "thought", "content": thought})
            steps.append({"type": "final_answer", "content": final})
            if on_step:
                on_step(step, "thought", thought)
                on_step(step, "final_answer", final)
            break

        # Parse action
        action_name, action_input = parse_action(reply)

        # Extract thought
        thought = ""
        if "Thought:" in reply:
            thought = reply.split("Thought:")[1]
            if "Action:" in thought:
                thought = thought.split("Action:")[0].strip()

        steps.append({"type": "thought", "content": thought})
        if on_step:
            on_step(step, "thought", thought)

        if action_name and action_name in TOOL_FUNCTIONS:
            steps.append({"type": "action", "tool": action_name, "input": action_input})
            if on_step:
                on_step(step, "action", f"{action_name}({action_input})")

            tool_fn = TOOL_FUNCTIONS[action_name]
            observation = tool_fn(action_input)

            steps.append({"type": "observation", "content": observation})
            if on_step:
                on_step(step, "observation", observation[:500])

            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            steps.append({"type": "error", "content": f"Unknown action: {action_name}"})
            messages.append({"role": "user", "content": "Please use one of the available tools or provide a Final Answer."})

    return steps
