"""Intent taxonomy for tau-bench retail, and the routing table intent -> (write tools, wiki sections).

Design notes
- Labels are derived from the *write* tools in retail (the only actions that change DB state).
- Read-only / identity tools are shared by every route: every task starts with auth + lookups.
- `info_query`: user only asks about orders/products (no write action in gold).
- `transfer`: gold contains transfer_to_human_agents.
- `out_of_scope`: request the retail agent must not serve -> escalate / refuse. Mostly from external data.
- `unclear`: not enough signal in the utterance (e.g. "I need help with an order") -> fallback to full prompt.
  Never a training label from gold actions; it comes from the confidence threshold or vague-utterance data.
"""

WRITE_INTENTS = [
    "cancel",
    "modify_items",
    "modify_order_address",
    "modify_payment",
    "modify_user_address",
    "return",
    "exchange",
]
INTENTS = WRITE_INTENTS + ["info_query", "transfer", "out_of_scope", "continue"]
# "continue": no new intent in this turn (identity details, confirmations, vague openers). In cumulative
# routing it keeps the current route: adds no tools and does NOT trigger fallback.
FALLBACK = "unclear"  # routing outcome, not a classifier class

ACTION_TO_INTENT = {
    "cancel_pending_order": "cancel",
    "modify_pending_order_items": "modify_items",
    "modify_pending_order_address": "modify_order_address",
    "modify_pending_order_payment": "modify_payment",
    "modify_user_address": "modify_user_address",
    "return_delivered_order_items": "return",
    "exchange_delivered_order_items": "exchange",
    "transfer_to_human_agents": "transfer",
}

SHARED_TOOLS = [
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "list_all_product_types",
    "calculate",
    "think",
    "transfer_to_human_agents",  # always available: it is also the safety valve
]

INTENT_WRITE_TOOLS = {i: [a for a, x in ACTION_TO_INTENT.items() if x == i] for i in WRITE_INTENTS}
INTENT_WRITE_TOOLS.update({"info_query": [], "transfer": [], "out_of_scope": [], "continue": []})

# wiki.md headings that each route needs on top of the always-kept preamble + "Domain basic"
INTENT_WIKI_SECTIONS = {
    "cancel": ["## Cancel pending order"],
    "modify_items": ["## Modify pending order", "### Modify items"],
    "modify_order_address": ["## Modify pending order"],
    "modify_payment": ["## Modify pending order", "### Modify payment"],
    "modify_user_address": [],
    "return": ["## Return delivered order"],
    "exchange": ["## Exchange delivered order"],
    "info_query": [],
    "transfer": [],
    "out_of_scope": [],
    "continue": [],
}


def intents_from_actions(actions):
    """Multi-label: set of intents implied by a gold action list. Empty write set -> info_query."""
    labels = []
    for a in actions:
        it = ACTION_TO_INTENT.get(a["name"])
        if it and it not in labels:
            labels.append(it)
    return labels or ["info_query"]


# Intents the text often cannot tell apart, because the difference lives in the database, not in the words:
#   modify_items vs exchange       -> order status (pending vs delivered), which only get_order_details reveals
#   order vs account address       -> "my address" is ambiguous until the agent looks at the user
# Routing one member adds the whole group (~300-450 extra tokens), which is far cheaper than a missing tool.
ROUTE_GROUPS = [{"modify_items", "exchange"}, {"modify_order_address", "modify_user_address"}]


def expand_groups(intents):
    s = set(intents)
    for g in ROUTE_GROUPS:
        if s & g:
            s |= g
    return sorted(s)


def needed_write_tools(intents):
    """Write tools a set of intents requires (shared read-only tools are always present)."""
    return {t for i in intents for t in INTENT_WRITE_TOOLS.get(i, [])}


def tools_for(intents, all_tool_names=None):
    names = list(SHARED_TOOLS)
    for i in intents:
        for t in INTENT_WRITE_TOOLS.get(i, []):
            if t not in names:
                names.append(t)
    return names
