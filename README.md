# Ai-ChatBot
# BVRIT College FAQ Chatbot 🎓

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
**BVRIT Hyderabad College of Engineering for Women** using a curated
knowledge base — with grounded citations, function calling for exact
lookups, persistent memory, observability, and governance guardrails.

Built as a full-stack LLM application: not just a chat wrapper, but a
system with retrieval, structured tool use, session memory, request
logging, and safety controls.

---

## ✨ Features

- **Grounded RAG answers** — every response is generated only from the
  curated knowledge document, never the model's general training data,
  with `[Section, Page]` citations on every claim
- **Graceful refusal** — the bot says so, with a fallback contact, when a
  question falls outside the knowledge base, instead of hallucinating
- **Function calling** — exact fee and admission-deadline lookups are
  answered by structured tools reading from JSON data, not LLM guesses
- **Persistent memory** — conversations are summarized and remembered
  across sessions via a "Remember me" identifier
- **Observability dashboard** — live request volume, latency, token cost
  estimates, refusal rate, and top questions
- **Governance & safety** — PII redaction before logging, per-session
  rate limiting, an advice-request filter, and a password-gated admin
  panel with CSV export and a maintenance kill switch
- **8-dimension automated evaluation suite** — functional, quality,
  safety, security, robustness, performance, context, and RAGAS
  (faithfulness, answer relevancy, context precision, context recall)
  metrics, with a results dashboard and a "weakest dimension" diagnosis
- **Light, warm UI** — custom design system (no dark theme, no stark
  white), college photos on the welcome screen, citation badges, and a
  typing indicator

---

## 🖼️ Screenshots

*(Add screenshots here once your UI is finalized — drag PNG/JPG files into
this section on GitHub, or reference them like:)*

```markdown
![Chat interface](assets/screenshots/chat.png)
![Evaluation dashboard](assets/screenshots/eval_dashboard.png)
```

---

## 🏗️ Architecture

```
User Question
     │
     ▼
Streamlit Chat UI
     │
     ▼
Governance layer (rate limit, PII redaction, advice filter)
     │
     ▼
LLM decides: structured tool call  OR  RAG retrieval
     │                                        │
     ▼                                        ▼
tools/ (fees.json, deadlines.json)   ChromaDB retriever (top-k chunks)
     │                                        │
     └───────────────┬────────────────────────┘
                      ▼
        Grounded generation (with citations)
                      │
                      ▼
        Answer + citations → Chat UI
                      │
                      ▼
        Logged to observability.db + memory.db
```

Offline indexing step:
```
data/college_info.docx → chunk → embed → ChromaDB (persisted)
```

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| Vector store | ChromaDB (persisted locally) |
| Embeddings | `text-embedding-3-small` (via OpenRouter) |
| Generation | GPT-4o Mini (via OpenRouter, swappable) |
| UI | Streamlit |
| Memory / Observability | SQLite |
| Evaluation | RAGAS + custom LLM-as-judge |

---

## 📁 Project Structure

```
Ai-ChatBot/
├── app.py                          # Streamlit chat UI
├── ingest.py                       # builds the vector index
├── rag_core/                       # retrieval + grounded generation
├── tools/                          # function-calling schemas + handlers
├── memory/                         # persistent session memory
├── observability/                  # request logging
├── governance/                     # guardrails, PII redaction, rate limits
├── evaluation/                     # 8-dimension test suite + RAGAS
├── pages/                          # Evaluation & Observability dashboards
├── data/                           # college_info.docx, fees.json, deadlines.json
├── assets/images/                  # logo, campus photos
├── spec.md                         # full technical specification
├── requirements.txt
└── .env.example
```

See [`spec.md`](./spec.md) for the full technical specification.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key with credits

### Setup

```bash
git clone https://github.com/Vyshnavi-279/Ai-ChatBot.git
cd Ai-ChatBot

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your own OPENROUTER_API_KEY and ADMIN_PASSWORD
```

### Build the knowledge index

```bash
python ingest.py
```

### Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Run the evaluation suite

```bash
python evaluation/generate_tests.py
python evaluation/run_tests.py
python evaluation/judge.py
python evaluation/ragas_eval.py
python evaluation/report.py
```

Results appear in `evaluation_report.json` and on the **Evaluation
Dashboard** page inside the app.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Your own key — never commit this |
| `ADMIN_PASSWORD` | Yes | Password for the sidebar admin panel — set your own value |
| `GENERATION_MODEL` | No | Default: `openai/gpt-4o-mini` |
| `JUDGE_MODEL` | No | Default: a different model than generation, to avoid self-bias |
| `EMBEDDING_MODEL` | No | Default: `openai/text-embedding-3-small` |
| `MAX_TOKENS` | No | Default: `300` |
| `TOP_K` | No | Default: `5` |

`.env` is gitignored — copy `.env.example` and fill in your own values.
Never commit real API keys or passwords.

---

## 📊 Evaluation Summary

The chatbot is tested against 20 automated test cases across 8 dimensions.
See the live results on the **Evaluation Dashboard** page, or the raw
output in `evaluation_report.json` after running the suite. The report
includes a pass/fail breakdown per dimension, RAGAS scores
(faithfulness, answer relevancy, context precision, context recall), the
weakest dimension, and a concrete recommended fix.

---

## ⚠️ Known Limitations

- The knowledge base document is manually curated from BVRIT's public
  website; some sections (e.g. detailed fee breakdowns, individual
  faculty profiles) may be incomplete where source data wasn't publicly
  available — the bot will refuse rather than guess in these cases.
- Free-tier or low-balance OpenRouter accounts may hit rate/credit
  limits during heavy testing (e.g. running the full evaluation suite).

---

## 📄 License

This project was built for educational purposes as part of a college
assignment. College information is sourced from BVRIT Hyderabad's public
website for demonstration purposes only.

---

## 🙋 Author

**Vyshnavi** — [github.com/Vyshnavi-279](https://github.com/Vyshnavi-279)
