import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from agent import HomeworkAgent


class ChatUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SmartTutor - Homework Agent")
        self.root.geometry("900x650")

        self.model_var = tk.StringVar(value=os.getenv("DASHSCOPE_MODEL", "deepseek-r1"))

        top = ttk.Frame(root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Model").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.model_var, width=28).pack(side=tk.LEFT, padx=(8, 16))

        self.btn_new = ttk.Button(top, text="New Chat", command=self.new_chat)
        self.btn_new.pack(side=tk.LEFT)

        self.btn_summary = ttk.Button(top, text="Summarize", command=self.on_summarize)
        self.btn_summary.pack(side=tk.LEFT, padx=8)

        self.btn_send = ttk.Button(top, text="Send", command=self.on_send)
        self.btn_send.pack(side=tk.RIGHT)

        mid = ttk.Frame(root, padding=(10, 0, 10, 10))
        mid.pack(fill=tk.BOTH, expand=True)

        self.chat = scrolledtext.ScrolledText(mid, wrap=tk.WORD, state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)

        self.input = ttk.Entry(bottom)
        self.input.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self.input.bind("<Return>", lambda _e: self.on_send())

        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(root, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.pack(fill=tk.X)

        self.agent = None
        self._busy = False

        self.append("Assistant", "Welcome to SmartTutor, your personal math and history homework tutor. What can I help you today?")
        self.new_chat()

    def set_busy(self, busy: bool, status: str):
        self._busy = busy
        self.status_var.set(status)
        state = tk.DISABLED if busy else tk.NORMAL
        self.btn_send.config(state=state)
        self.btn_new.config(state=state)
        self.btn_summary.config(state=state)
        self.input.config(state=state)

    def append(self, who: str, text: str):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{who}: {text}\n\n")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def new_chat(self):
        model = self.model_var.get().strip() or "deepseek-r1"
        try:
            self.agent = HomeworkAgent(model=model)
        except SystemExit:
            messagebox.showerror(
                "Missing API Key",
                "Environment variable DASHSCOPE_API_KEY not found.\n\n"
                "In PowerShell, set it like:\n"
                "$env:DASHSCOPE_API_KEY=\"<your_key>\"",
            )
            return
        self.append("System", f"New chat started. Model = {model}")

    def on_send(self):
        if self._busy:
            return
        text = self.input.get().strip()
        if not text:
            return
        self.input.delete(0, tk.END)
        self.append("User", text)
        self.run_agent_call(lambda: self.agent.ask(text, stream=False), title="Assistant")

    def on_summarize(self):
        if self._busy:
            return
        self.append("User", "Can you summarise our conversation so far?")
        self.run_agent_call(lambda: self.agent.summarize(), title="Assistant (summary)")

    def run_agent_call(self, fn, title: str):
        self.set_busy(True, "Calling API...")

        def worker():
            try:
                result = fn()
            except Exception as e:
                result = f"Sorry, request failed: {e}"

            def done():
                self.append(title, result)
                self.set_busy(False, "Ready.")

            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    ChatUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

