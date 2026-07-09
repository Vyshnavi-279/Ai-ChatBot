import streamlit as st

# Ensure this is the first Streamlit command in your script!
st.set_page_config(layout="wide", initial_sidebar_state="auto")

def inject_global_css():
    st.markdown("""
    <style>
        /* 1. Global Light Theme Background & Text */
        .stApp {
            background-color: #F5F1EA !important;
            color: #293241 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* 2. Sidebar Restyling */
        [data-testid="stSidebar"] {
            background-color: #EDE7DC !important;
            border-right: 1px solid #E4E0D8;
            padding: 24px 16px !important; /* 8px grid alignment */
        }
        
        /* 3. Consistent 8px Spacing Grid for Cards, Blocks, and Chat Container */
        .block-container {
            padding-top: 32px !important;
            padding-bottom: 32px !important;
            padding-left: 24px !important;
            padding-right: 24px !important;
            gap: 16px !important;
        }
        
        /* 4. Chat Bubbles Design System */
        .user-bubble {
            background-color: #98C1D9 !important; /* --accent */
            color: #293241 !important;
            padding: 16px;
            border-radius: 12px 12px 0px 12px;
            margin-bottom: 16px;
            margin-left: auto;
            max-width: 80%;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
            text-align: right;
        }
        
        .bot-bubble {
            background-color: #FFFFFF !important; /* --surface */
            border: 1px solid #E4E0D8;
            color: #293241 !important;
            padding: 16px;
            border-radius: 12px 12px 12px 0px;
            margin-bottom: 16px;
            max-width: 80%;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
        }

        /* 5. Custom Citation and Refusal Badges */
        .citation-badge {
            display: inline-block;
            background-color: rgba(152, 193, 217, 0.2); /* Tinted --accent */
            color: #3D5A80 !important; /* --primary */
            font-size: 0.75rem;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 8px;
            margin: 4px 4px 4px 0px;
            border: 1px solid #98C1D9;
        }
        
        .refused-badge {
            display: inline-block;
            background-color: rgba(188, 71, 73, 0.1);
            color: #BC4749 !important; /* --error */
            font-size: 0.75rem;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 8px;
            border: 1px solid #BC4749;
            margin-bottom: 8px;
        }

        /* 6. Subtle Micro-interactions & Hover States */
        div.stButton > button {
            background-color: #3D5A80 !important; /* --primary */
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 8px 16px !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        div.stButton > button:hover {
            background-color: #293241 !important;
            transform: translateY(-1px);
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Question chips hover styling */
        .question-chip {
            background-color: #FFFFFF;
            border: 1px solid #E4E0D8;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s ease;
            margin: 4px;
            display: inline-block;
        }
        .question-chip:hover {
            background-color: rgba(152, 193, 217, 0.15) !important;
            border-color: #98C1D9;
            transform: scale(1.02);
        }

        /* 7. Typing Indicator Animation */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 8px;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            background-color: #98C1D9;
            border-radius: 50%;
            animation: pulse 1.4s infinite ease-in-out both;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes pulse {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1.0); }
        }
        
        /* 8. Mobile Breakpoint Fallback Adjustments */
        @media (max-width: 768px) {
            .user-bubble, .bot-bubble { max-width: 100% !important; }
            .block-container { padding: 16px !important; }
        }
    </style>
    """, unsafe_html=True)

inject_global_css()
import streamlit as st
import pandas as pd

# Load the data from the database
df = pd.read_sql_query("SELECT * FROM observability", conn)

# Create a line chart of requests over time
line_chart = st.line_chart(df['timestamp'], df['requests'])

# Create a dashboard with various charts and tables
st.sidebar.header('Observability Dashboard')
st.sidebar.sidebar(
    st.sidebar.selectbox('Metric', ['Requests', 'Average Latency', 'P95 Latency'])
)
st.sidebar.sidebar(
    st.sidebar.selectbox('Visualization', ['Line Chart'])
)
st.sidebar.sidebar(
    st.sidebar.button('Download CSV')
)
st.sidebar.sidebar(
    st.sidebar.button('Maintenance Mode')
)

# Generate the chart and display it
streamlit.sidebar.add_layout(line_chart)
