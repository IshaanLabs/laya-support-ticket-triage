"""Prebuilt example tickets for the Live demo — one-click, no typing.

Chosen to show the range of behaviors: a clear high-confidence route, an
urgent churn-risk case, an ambiguous ticket the model should be unsure about,
and a refund/returns case. Each is a realistic support message.
"""

EXAMPLES = [
    {
        "label": "Billing dispute + churn threat",
        "text": (
            "I was charged twice for my March invoice (#4411). This is the second month "
            "this has happened. Please refund the duplicate charge today or I will cancel "
            "our subscription and move to a competitor."
        ),
        "note": "Clear billing signal + explicit cancel threat — expect high-confidence Billing route, high churn.",
    },
    {
        "label": "Production outage (urgent)",
        "text": (
            "Our entire team has been unable to access the platform for the last two hours. "
            "The dashboard won't load and the API returns 503 errors. This is blocking all of "
            "our operations. We need an update on the outage immediately."
        ),
        "note": "Outage language — expect Service Outages / Technical, high urgency.",
    },
    {
        "label": "Ambiguous integration question",
        "text": (
            "Hello, I'd like some guidance on connecting your product with our internal tools. "
            "I'm not entirely sure which options are available or where to start. Could you point "
            "me in the right direction?"
        ),
        "note": "Vague — could be Product/Technical/Customer Service. Expect LOW confidence -> escalate.",
    },
    {
        "label": "Refund / return request",
        "text": (
            "The device I purchased last week doesn't meet my needs. I'd like to return it and "
            "get a refund. Can you tell me the return process and whether I'm within the window?"
        ),
        "note": "Returns/refund intent — expect Returns and Exchanges, high refund signal.",
    },
    {
        "label": "Login / account access issue",
        "text": (
            "I keep getting 'invalid credentials' when logging into my account, even after "
            "resetting my password twice. I've tried three different browsers. Please help me "
            "regain access."
        ),
        "note": "Account/technical — expect IT/Technical Support.",
    },
]
