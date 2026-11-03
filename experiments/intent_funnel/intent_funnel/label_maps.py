"""External dataset label -> tau-bench-retail intent. None = excluded (ambiguous mapping).

Principle: only map when a retail agent with tau-bench tools would take that route. Requests the retail
agent has no tool for become `out_of_scope` (these are the valuable in-domain-but-unsupported negatives).
"""

ABCD_SUBFLOW = {
    # writes
    "manage_cancel": "cancel",
    "manage_change_address": "modify_user_address",   # noisy: some users mean an order address
    "manage_payment_method": "modify_payment",        # noisy: account-level vs order-level payment
    "status_payment_method": "modify_payment",        # "payment method on my order is wrong"
    "refund_initiate": "return",
    "return_stain": "return",
    "return_color": "return",
    "return_size": "return",
    # read-only lookups
    "status": "info_query",                # shipping_issue/status
    "missing": "info_query",
    "status_delivery_time": "info_query",
    "status_quantity": "info_query",
    "refund_status": "info_query",
    "refund_update": "info_query",
    "boots": "info_query", "shirt": "info_query", "jeans": "info_query", "jacket": "info_query",
    # excluded (mixed or no clean retail equivalent)
    "manage": None,          # shipping_issue/manage mixes cancel / address / wrong item
    "manage_create": None,   # "add an item" — retail can only swap same-product items
    "cost": None,
}
# subflows under these flows the retail agent cannot serve at all
ABCD_OOS_FLOWS = {"account_access", "purchase_dispute", "subscription_inquiry", "troubleshoot_site",
                  "storewide_query"}
ABCD_OOS_SUBFLOWS = {"status_service_added", "status_service_removed", "status_shipping_question",
                     "status_credit_missing", "manage_change_name", "manage_change_phone",
                     "status_mystery_fee", "manage_upgrade", "manage_downgrade"}


def abcd_label(flow, subflow):
    if flow in ABCD_OOS_FLOWS or subflow in ABCD_OOS_SUBFLOWS:
        return "out_of_scope"
    return ABCD_SUBFLOW.get(subflow)


BITEXT_INTENT = {
    "cancel_order": "cancel",
    "change_order": "modify_items",
    "change_shipping_address": "modify_order_address",
    "set_up_shipping_address": "modify_user_address",
    "get_refund": "return",
    "track_order": "info_query",
    "track_refund": "info_query",
    "delivery_period": "info_query",
    "delivery_options": "info_query",
    "check_payment_methods": "info_query",
    "check_refund_policy": "info_query",
    "check_cancellation_fee": "info_query",
    "contact_human_agent": "transfer",
    "contact_customer_service": "transfer",
    "complaint": None,
    # account / billing / marketing features retail tools don't cover
    "check_invoice": "out_of_scope", "get_invoice": "out_of_scope", "edit_account": "out_of_scope",
    "switch_account": "out_of_scope", "create_account": "out_of_scope", "delete_account": "out_of_scope",
    "recover_password": "out_of_scope", "registration_problems": "out_of_scope",
    "newsletter_subscription": "out_of_scope", "review": "out_of_scope", "place_order": "out_of_scope",
    "payment_issue": "out_of_scope",
}
