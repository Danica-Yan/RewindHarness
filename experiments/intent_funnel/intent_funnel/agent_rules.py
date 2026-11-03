"""Agent-proposal rules: which write intents does the agent's last message propose?

The keyword patterns are the weak-labelling MENTION patterns, unchanged. They are applied only to
proposal/confirmation sentences ("would you like…", "shall I…", "please confirm…"), so policy chatter
such as "this order cannot be cancelled" doesn't add tools. Shared by the offline evaluation
(scripts/agent_rules_heldout.py) and the live router (rewind_router.py).
"""
import re

from .taxonomy import WRITE_INTENTS

MENTION = {
    "cancel": r"\bcancel",
    "exchange": r"\bexchang|\bswap|\breplace|\bswitch\b|\bdifferent (one|model|version|color|size)",
    "return": r"\breturn|\brefund|\bsend (it|them|.*) back",
    "modify_items": r"\b(change|modify|modif|switch|swap|update|different|instead)\b",
    "modify_order_address": r"\baddress|\bmoved\b|\bship(ping)? to\b",
    "modify_user_address": r"\baddress|\bmoved\b",
    "modify_payment": r"\bpay|\bcard\b|\bpaypal|\bgift card",
    "transfer": r"\b(human|agent|representative|real person|someone|supervisor)\b",
}
MENTION = {k: re.compile(v, re.I) for k, v in MENTION.items()}
PROPOSAL = re.compile(r"\b(would you like|shall i|should i|do you want|please confirm|confirm (that|if|whether)|"
                      r"i can|we can|you can|let me|i'?ll|can i)\b", re.I)
SENT = re.compile(r"(?<=[.!?\n])\s+")


def proposal_text(agent_text: str) -> str:
    return " ".join(s for s in SENT.split(agent_text or "") if PROPOSAL.search(s))


def proposed_intents(agent_text: str, proposal_only: bool = True) -> list[str]:
    text = proposal_text(agent_text) if proposal_only else (agent_text or "")
    return [c for c in WRITE_INTENTS if text and MENTION[c].search(text)]
