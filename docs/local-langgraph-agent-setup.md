# Local LangGraph Agent with Ollama over Tailscale

A setup guide for running a LangChain/LangGraph ReAct agent against a self-hosted Ollama LLM, with the notebook environment and inference server on separate machines connected via Tailscale.

## What this is

A web-browsing Q&A agent that:

- Runs in a Jupyter notebook on a local dev machine
- Uses LangGraph's `create_react_agent` with conversation memory (via `MemorySaver`)
- Calls a local LLM (llama3.1) hosted on Ollama on a different machine
- Reaches Ollama privately over Tailscale — no public exposure, no API keys

This replaces a Colab + OpenAI version of the same notebook. The migration eliminates per-call API costs and keeps everything on your own hardware.

## Architecture

```
┌─────────────────┐         Tailscale          ┌──────────────────┐
│  Dev Machine    │  ◄───────VPN───────►       │  Ollama Host     │
│                 │                            │                  │
│  JupyterLab     │   HTTP :11434              │  ollama serve    │
│  Python venv    │ ─────────────────────►     │  llama3.1 model  │
│  LangGraph code │                            │                  │
└─────────────────┘                            └──────────────────┘
```

The notebook code calls `ChatOllama(base_url="http://<tailscale-ip>:11434", ...)`. Tailscale handles the routing; neither machine needs to be on the same LAN or have public ports open.

## Prerequisites

- Two machines (can be the same machine if you want — skip Tailscale in that case)
- A Tailscale account, with both machines logged in
- An Ollama-supported OS on the inference host (macOS, Linux, or Windows)

## Part 1: Set up the Ollama host

This is the machine that will run the model. Ideally something with decent RAM (16GB+ for 8B models) and, optionally, a GPU.

### Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Or download the installer from ollama.com.

### Configure Ollama to listen on the network

By default Ollama only accepts connections from `localhost`. To accept connections from other Tailscale nodes, set `OLLAMA_HOST` before starting it:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

To make this permanent on Linux (systemd):

```bash
sudo systemctl edit ollama.service
```

Add:

```
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then `sudo systemctl restart ollama`.

### Pull a model

For a tool-calling agent like this one, pick a model that handles tool calls reliably. `llama3.1` (the 8B default) works well. `qwen2.5` and `mistral-nemo` are also solid. Avoid small models (sub-7B) and pure embedding models like `nomic-embed-text` — the agent loop needs a chat model that can emit structured tool calls.

```bash
ollama pull llama3.1
```

This downloads about 4.9GB. On a fast connection it takes a few minutes; on slow or metered connections, plan accordingly. The model is cached locally afterward (in `~/.ollama/models` on macOS/Linux), so you only pay this cost once.

Verify what's installed:

```bash
ollama list
```

**Important: the tag you pull must match the `model="..."` string in the notebook.** Ollama treats `llama3.1` and `llama3.1:70b` as different models. If you pull `llama3.1:70b` here, you need `model="llama3.1:70b"` in Steps 4 and 5 of the notebook — otherwise you'll get a 404 "model not found" error at runtime even though Ollama is responding fine. The bare name `llama3.1` is shorthand for `llama3.1:latest`, which currently resolves to the 8B variant.

### Find the Tailscale IP

```bash
tailscale ip -4
```

Note this address — you'll need it from the dev machine. Tailscale IPs are in the `100.x.x.x` range.

### Smoke test from the Ollama host itself

```bash
curl http://localhost:11434/api/tags
```

Should return JSON listing the models you've pulled.

## Part 2: Set up the dev machine

This is where Jupyter and the notebook will live.

### Install Python (if needed)

macOS ships with Python 3.9 from the Xcode Command Line Tools, but it's old enough that some packages have started dropping support. Install a current version:

```bash
# macOS
brew install python

# Ubuntu/Debian
sudo apt install python3 python3-pip python3-venv
```

### Create a virtual environment

Don't install Jupyter system-wide. Use a venv so notebook dependencies stay isolated:

```bash
python3 -m venv ~/jupyter-env
source ~/jupyter-env/bin/activate
```

Add an alias to `~/.zshrc` (or `~/.bashrc`) to make activation easier:

```bash
alias jupyenv="source ~/jupyter-env/bin/activate"
```

### Install JupyterLab

With the venv activated:

```bash
pip install --upgrade pip
pip install jupyterlab
```

### Install the notebook's runtime dependencies

```bash
pip install langchain langchain-openai langchain-community langgraph langchain-ollama requests beautifulsoup4
```

(`langchain-openai` isn't strictly required for the local-only path, but it's harmless and makes it easy to swap providers later.)

### Verify Tailscale routing

From the dev machine, before launching anything:

```bash
curl http://<ollama-tailscale-ip>:11434/api/tags
```

If this returns JSON, the network path works. If it hangs or refuses, debug Tailscale before going further (see Troubleshooting).

### Launch JupyterLab

Make sure the venv is activated first (your prompt should show `(jupyter-env)`):

```bash
source ~/jupyter-env/bin/activate    # or use the `jupyenv` alias
jupyter lab --ip=0.0.0.0 --no-browser
```

The `--ip=0.0.0.0` lets you reach JupyterLab from any device on your Tailnet. Without it, JupyterLab binds to localhost only and other machines won't be able to connect.

### Find the token in the launch output

When JupyterLab starts, it prints several lines of startup info, ending with a block like this:

```
    To access the server, open this file in a browser:
        file:///Users/you/Library/Jupyter/runtime/jpserver-12345-open.html
    Or copy and paste one of these URLs:
        http://<dev-machine>:8888/lab?token=c8de56fa1234abcd5678ef90...
        http://127.0.0.1:8888/lab?token=c8de56fa1234abcd5678ef90...
```

The long string after `token=` is what authenticates you. You have two options:

**Option A — Use the full URL.** Copy the `http://<hostname>:8888/lab?token=...` line directly and paste into your browser. It logs you in automatically.

**Option B — Visit the bare URL and paste the token.** Go to `http://<hostname>:8888/lab` (e.g. `http://<dev-machine>:8888/lab`) and paste just the token portion (everything after `token=`) into the "Password or token" field.

**If you missed the output** (closed the terminal, scrolled past it, etc.), open a new terminal on the same machine, activate the venv, and run:

```bash
source ~/jupyter-env/bin/activate
jupyter server list
```

This prints the running server with its token in the URL.

### Open JupyterLab in a browser

From any machine on your Tailnet (your laptop, phone, whatever), navigate to:

```
http://<dev-machine-tailscale-name>:8888/lab
```

For example: `http://<dev-machine>:8888/lab` (using the Tailscale MagicDNS hostname) or `http://100.x.x.x:8888/lab` (using the raw Tailscale IP from `tailscale ip -4`).

Authenticate with the token from the previous step. Once you're in, you can upload a `.ipynb` file via drag-and-drop into the file browser on the left, or with the upload button at the top of that panel.

### Operational tips

- **Keep the terminal open.** Closing it kills the server. To detach, run inside `tmux` or `screen` (see Operational notes at the bottom).
- **Set a password instead of using tokens.** After logging in once, you can run `jupyter lab password` (in another terminal, with the venv activated) to set a persistent password — easier than hunting for tokens after each restart.
- **Skip auth entirely on a trusted Tailnet.** Add `--ServerApp.token='' --ServerApp.password=''` to the launch command. This is reasonable *because* Tailscale already gates who can reach port 8888 — don't do this on a publicly exposed server.

## Part 3: The notebook

The notebook has six logical sections. Here are the cells in the order they should appear, with the Ollama-specific bits highlighted.

### Step 1 — Install packages (idempotent re-install from inside the notebook)

```python
!pip install -q langchain langchain-openai langchain-community langgraph langchain-ollama
```

### Step 2 — API key loading

**Skip this entirely.** Local Ollama doesn't need an API key. If you're adapting an OpenAI version of this notebook, comment out or delete the cell.

### Step 3 — Imports

```python
import os
import requests
from bs4 import BeautifulSoup

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
```

If you're porting from a Colab notebook, remove `from google.colab import userdata` — it only exists inside Colab.

### Step 4 — Define the website-fetching tool

```python
def extract_website_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def website_qa_tool(input_text):
    """Fetch a URL or answer from chat history."""
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
        base_url="http://<your-ollama-tailscale-ip>:11434",  # ← e.g. http://100.x.x.x:11434
    )

    if '|' in input_text and "<website URL>" not in input_text:
        url, question = input_text.split('|', 1)
        content = extract_website_text(url.strip())
        prompt = (
            f"Answer this question using the content from the website:\n\n"
            f"{question.strip()}\n\nContent:\n{content[:6000]}"
        )
    else:
        if "<website URL>" in input_text and "|" in input_text:
            _, question = input_text.split('|', 1)
            input_text = question.strip()
        prompt = f"Answer the user's question as best you can.\n\nQuestion:\n{input_text}"

    response = llm.invoke(prompt)
    return response.content

web_tool = Tool(
    name="WebsiteQnA",
    func=website_qa_tool,
    description=(
        "If input is '<website URL> | <question>' → fetch from website, or "
        "if input is '<question>' → answer using chat history."
    ),
)

memory = MemorySaver()
```

### Step 5 — Build the agent

```python
agent = create_react_agent(
    model=ChatOllama(
        model="llama3.1",
        temperature=0,
        base_url="http://<your-ollama-tailscale-ip>:11434",  # ← same IP as Step 4
    ),
    tools=[web_tool],
    checkpointer=memory,
)
```

### Step 6 — Ask questions

```python
config = {"configurable": {"thread_id": "etrade-demo"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="https://us.etrade.com | Can you tell me what this website offers to beginner investors?")]},
    config=config,
)
print(response["messages"][-1].content)
```

Follow-up questions reuse the same `thread_id` so memory persists:

```python
response = agent.invoke(
    {"messages": [HumanMessage(content="Based on what you told me earlier, what kinds of trading options are available for active traders?")]},
    config=config,
)
print(response["messages"][-1].content)
```

## Tip: Define the Ollama URL once

Rather than repeating the IP across both `ChatOllama` calls, set it once at the top of the notebook:

```python
OLLAMA_URL = "http://<your-ollama-tailscale-ip>:11434"
OLLAMA_MODEL = "llama3.1"
```

Then reference `base_url=OLLAMA_URL, model=OLLAMA_MODEL` everywhere. Saves a search-and-replace when you switch hosts or models.

## Troubleshooting

These are the actual errors I hit during setup, in roughly the order they appeared.

### `OpenAIError: The api_key client option must be set...`

You're still instantiating `ChatOpenAI` somewhere. Search the whole notebook for `ChatOpenAI` and `OpenAI` — every hit needs to be `ChatOllama` instead. Common miss: a tool function that creates its own LLM internally (Step 4 above). Re-run any edited cells after fixing.

### `ConnectError: [Errno 111] Connection refused`

The host is reachable but nothing's listening on port 11434. Either Ollama isn't running, or it's bound to localhost only. Check on the Ollama host:

```bash
ollama list                   # is it running?
ss -tlnp | grep 11434         # what is it listening on?
```

If it's bound to `127.0.0.1:11434`, restart with `OLLAMA_HOST=0.0.0.0:11434`.

### `ConnectTimeout: [Errno 110] Connection timed out`

The host isn't reachable at all. Two common causes:

1. **You're on Colab and Ollama is on a private network.** Colab can't reach Tailscale IPs. Move the notebook to a machine on your Tailnet, or expose Ollama publicly via Tailscale Funnel or ngrok.
2. **Tailscale isn't routing.** From the dev machine, `tailscale ping <host>` to confirm the path. If that fails, restart Tailscale (`tailscale down && tailscale up`).

### `ResponseError: model 'llama3.1' not found (status code: 404)`

Good news: you're talking to Ollama. Bad news: that model isn't pulled on it. SSH to the Ollama host and run:

```bash
ollama pull llama3.1
ollama list  # verify
```

Or change the `model="..."` argument to match a model you already have.

### `ModuleNotFoundError: No module named 'google'`

You're running a notebook ported from Colab outside Colab. Comment out or delete `from google.colab import userdata`.

### `NameError: name 'config' is not defined`

The kernel was restarted (or you skipped earlier cells). Variables don't survive kernel restarts. Either re-run from the top, or define `config` inline in the cell that needs it.

### `zsh: command not found: jupyter` (in a fresh terminal)

Each new shell starts without the venv activated. Run `source ~/jupyter-env/bin/activate` (or your `jupyenv` alias) first.

### How to confirm requests are actually hitting Ollama

Three quick checks, in order of effort:

1. **Error type.** `ResponseError` with a status code = the server replied. `ConnectError`/`ConnectTimeout` = it didn't.
2. **Direct curl.** `curl http://<host>:11434/api/tags` from the dev machine. JSON response = path works.
3. **Server logs.** On the Ollama host, `journalctl -u ollama -f` (systemd) or watch the terminal where it's running. Live request logs confirm traffic.

## Operational notes

### Keeping Jupyter running after you log out

If you SSH'd into the dev machine to launch Jupyter, closing the SSH session kills the server. Use `tmux`:

```bash
tmux new -s jupyter
source ~/jupyter-env/bin/activate
jupyter lab --ip=0.0.0.0 --no-browser
# detach: Ctrl+b then d
# reattach later: tmux attach -t jupyter
```

### Switching models

Pull the new model on the Ollama host (`ollama pull qwen2.5`), update the `model="..."` strings in Steps 4 and 5, re-run those cells. No restart needed.

### First call is slow

Ollama lazy-loads models into memory on first request. Expect 10–30 seconds the first time after `ollama serve` starts; subsequent calls are fast until the model is unloaded (which happens after some idle time by default).

### Watching the agent reason

To see the full ReAct trace (thought → tool call → tool result → answer), print the whole message list, not just the last one:

```python
for msg in response["messages"]:
    print(type(msg).__name__, "::", msg.content[:200])
```

Useful for debugging and for understanding what `create_react_agent` is actually doing under the hood.
