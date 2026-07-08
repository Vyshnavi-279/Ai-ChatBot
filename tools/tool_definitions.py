# tools/tool_definitions.py
# OpenAI-style function-calling schemas for the four tools in spec.md §11.
# Tools return structured JSON data, never free text — the LLM formats the
# final answer, but the number/date itself comes from the tool.

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_fee_by_branch",
            "description": (
                "Get the exact fee for a specific branch/program at BVRIT from "
                "structured data (data/fees.json). Use this for any precise "
                "fee/tuition lookup instead of the knowledge-base retrieval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": (
                            "The branch or program name, e.g. 'CSE', 'ECE', "
                            "'EEE', 'CSE (AI&ML)', 'M.Tech (Data Sciences)'."
                        ),
                    }
                },
                "required": ["branch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_admission_deadline",
            "description": (
                "Get the exact admission deadline for a specific program at "
                "BVRIT from structured data (data/deadlines.json). Use this "
                "for any precise deadline/date lookup."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "program": {
                        "type": "string",
                        "description": (
                            "The program name, e.g. 'B.Tech (Category A)', "
                            "'B.Tech (Category B)', 'M.Tech (CSE)'."
                        ),
                    }
                },
                "required": ["program"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_unanswered_question",
            "description": (
                "Record a question the knowledge base could not answer, so the "
                "gap can be reviewed and the knowledge base improved. Call this "
                "whenever you must refuse because the answer is not in context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question that could not be answered.",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_contact",
            "description": (
                "Return the official BVRIT contact information (address, phone, "
                "email) and log the escalation. Use when the user needs to reach "
                "the college directly or their issue cannot be resolved here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the conversation is being escalated to a human contact.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]
