"""Clean user-simulator artefacts out of tau-bench user turns (DeepSeek as user simulator).

Two leak types seen in real runs:
  dsml_leak : DeepSeek's native tool-call markup (<｜｜DSML｜｜ calls> ...) inside the user's message
  role_leak : after the genuine user text, "\\n\\nuser" / "\\n\\nassistant" followed by a hallucinated
              agent reply (order summaries, "Shall I proceed?") — the simulator confused roles
The genuine prefix is kept; the turn is flagged so it can be excluded from training.
"""
import re

DSML = re.compile(r"</?｜｜DSML｜｜[^>]*>")
ROLE = re.compile(r"\n\s*(user|assistant)(?=\S)", re.M)


def clean(text):
    """-> (clean_text, flag or None)"""
    flag = None
    m = ROLE.search(text)
    if m:
        text, flag = text[:m.start()], "role_leak"
    if DSML.search(text):
        text, flag = DSML.sub("", text), flag or "dsml_leak"
    return text.strip(), flag
