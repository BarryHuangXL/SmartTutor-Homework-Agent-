# CSIT5900 Project 1 - SmartTutor (Homework Agent)

This project provides a **multi-turn homework tutoring agent** with:

- a simple **desktop chat window** (`tkinter`)
- a simple **web chat UI** + **FastAPI backend**

## Setup

### 1) Create a Python environment (recommended)

```bash
python -m venv .venv
```

On PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

If you are using a mirror that doesn't provide `openai>=1.x`, install `openai` from the official index:

```bash
python -m pip install -U --index-url https://pypi.org/simple openai
```

### 3) Set API key

PowerShell:

```powershell
$env:DASHSCOPE_API_KEY="<your_api_key>"
```

Optional model override (default is `deepseek-r1`):

```powershell
$env:DASHSCOPE_MODEL="deepseek-r1"
```

## Run

### Option A: Web UI (recommended for demo)

Start the API server:

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open the web page:

- `http://127.0.0.1:8000/`

### Option B: Desktop UI

```bash
python gui.py
```

### Option C: CLI
(If both Option A and Option B are incompatible with your computer environment, you can use Option C)

Interactive chat:

```bash
python agent.py
```

Single question:

```bash
python agent.py -q "Is square root of 1000 a rational number?"
```

In interactive mode, you can type `summary` / `summarize` to get a conversation summary.

