from transformers import AutoSchema
from transformers import pipelines

schema = AutoSchema(
    fields={
        "get_fee_by_branch": str,
        "get_admission_deadline": str,
        "log_unanswered_question": str,
        "escalate_to_contact": str,
    }
)
