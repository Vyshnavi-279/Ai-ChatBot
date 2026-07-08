import re

# Define a regex for PII redaction
PII_REGEX = r'\d{12,15}|\w{1,30}'

# Apply PII redaction to user input before logging or storing
def redact_pii(text):
    for pattern in PII_REGEX.split():
        text = re.sub(pattern, '***', text)
    return text

# Define rate limiter settings
RATE_LIMIT_DURATION = 10
RATE_LIMIT_THRESHOLD = 10

# Define functions to apply rate limiter and check for advice requests
def is_advice_request(question):
    # Check if the question contains sensitive information
    return re.search(r"medical|legal|financial", question)

def enforce_rate_limit(user_input):
    # Apply rate limiter and check for advice requests
    if not is_advice_request(user_input):
        return True
    return False

# Add rate limiter and check for advice request checks to the sidebar
st.sidebar.sidebar(
    st.sidebar.selectbox(
        'Rate Limit',
        ['Disable Rate Limiting', '10 Questions per Session'],
    )
)
st.sidebar.sidebar(
    st.sidebar.button('Apply Rate Limit')
)
