# rag_core/generation.py
# Grounding prompt (spec section 6) + LLM generation via OpenRouter.
# The generate() function returns both the cleaned answer text and a
# separate list of citation strings so the UI can render them as badges.

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from tools.tool_definitions import TOOL_DEFINITIONS
from tools.tool_handlers import TOOL_HANDLERS

from rag_core.config import (
    GENERATION_MODEL,
    MAX_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


# ---------------------------------------------------------------------------
# Grounding system prompt — implements all 5 rules from spec section 6.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are the official information assistant for BVRIT Hyderabad College of \
Engineering for Women. Your sole purpose is to answer questions about BVRIT \
using only the retrieved document excerpts provided to you.

RULES — follow every rule exactly, without exception:

1. GROUNDING RULE
   Answer ONLY from the retrieved context below. Never use your training \
knowledge, general world knowledge, or information not present in the \
provided excerpts. If the context does not contain enough information to \
answer, go to rule 4.

2. CITATION FORMAT
   After every factual claim, immediately append a citation in this exact \
format: [Section Name, Page N]
   Example: "The college was established in 2012 [About BVRIT, Page 1]."
   Use the section name and page number from the chunk metadata supplied \
with each excerpt. Every claim needs a citation — do not write uncited \
sentences.

3. CONFLICT HANDLING
   If two excerpts contradict each other on the same fact, present both \
versions and explicitly flag the discrepancy:
   "Source A states X [Section, Page N], but Source B states Y \
[Section, Page M]. Please verify with the college directly."

4. REFUSAL INSTRUCTION
   If the answer cannot be found in the retrieved context, respond with \
exactly this and nothing else:
   "I'm sorry, that information is not available in my knowledge base. \
For accurate and up-to-date details, please contact BVRIT Hyderabad \
directly at the contact details in our Contact section."
   Do not attempt to answer from memory. Do not guess or extrapolate.

5. TONE & FORMAT
   Be concise, factual, and professional. No marketing language, no \
superlatives, no invented details. Use bullet points for lists. Keep \
answers focused on what was asked.
"""


# ---------------------------------------------------------------------------
# Lazy singleton LLM client
# ---------------------------------------------------------------------------
_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=GENERATION_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=0.0,    # deterministic; factual Q&A needs no creativity
            max_tokens=MAX_TOKENS,
        )
    return _llm


# Cap for tool-augmented answers (spec: keep max_tokens conservative).
TOOL_ANSWER_MAX_TOKENS = 512

# Cap for conversation summarization (memory layer).
SUMMARY_MAX_TOKENS = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    if not chunks:
        return "No relevant excerpts were retrieved from the knowledge base."

    lines: list[str] = ["Retrieved excerpts (use these as your ONLY source):"]
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        section = meta.get("section", "Unknown Section")
        page = meta.get("page", "?")
        text = chunk.get("text", "").strip()
        lines.append(
            f"\n--- Excerpt {i} | Section: {section} | Page: {page} ---\n{text}"
        )
    return "\n".join(lines)


_CITATION_RE = re.compile(r"\[([^\[\]]+?,\s*Page\s*\d+)\]", re.IGNORECASE)


def _extract_citations(answer: str) -> tuple[str, list[str]]:
    """Return (cleaned_answer, unique_citation_list).

    The answer text is returned unchanged — citations stay inline so the
    full answer reads naturally.  The separate list lets the UI render
    badges without parsing the text again.
    """
    citations = list(dict.fromkeys(_CITATION_RE.findall(answer)))  # ordered-unique
    return answer, citations


def _is_refusal(answer: str) -> bool:
    """True if the LLM triggered the refusal instruction."""
    return "not available in my knowledge base" in answer.lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    query: str,
    chunks: list[dict],
    history: Optional[list[dict]] = None,
    user_context: Optional[str] = None,
) -> dict:
    """Call the generation LLM and return a structured result.

    Parameters
    ----------
    query:
        The user's current question.
    chunks:
        Retrieved document chunks from ``retriever.retrieve()``.
    history:
        Conversation history as a list of ``{"role": "user"|"assistant",
        "content": "..."}`` dicts, oldest first.  Pass ``None`` or ``[]``
        for a fresh conversation.
    user_context:
        Optional long-term memory summary for a returning user.  When set,
        it is prepended to the system prompt as "Returning user context".

    Returns
    -------
    dict with keys:
        ``answer``    — full answer text with inline citations
        ``citations`` — list of unique citation strings extracted from answer
        ``refused``   — True if the LLM triggered the refusal instruction
        ``model``     — model name used for generation
        ``tool_calls`` — list of tool names invoked (empty for pure RAG)
    """
    llm = _get_llm()
    context_block = _build_context_block(chunks)

    system_prompt = SYSTEM_PROMPT
    if user_context:
        system_prompt = f"Returning user context: {user_context}\n\n{system_prompt}"

    # Build message list: system → history → context-augmented user turn.
    messages: list = [SystemMessage(content=system_prompt)]

    for turn in (history or []):
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Current turn: inject retrieved context alongside the question.
    augmented_query = (
        f"{context_block}\n\n"
        f"User question: {query}"
    )
    messages.append(HumanMessage(content=augmented_query))

    # First pass: let the LLM decide between answering directly (RAG flow)
    # and requesting one of the structured tools (spec.md §11).
    llm_with_tools = llm.bind_tools(TOOL_DEFINITIONS)
    response = llm_with_tools.invoke(messages)

    tool_calls_made: list[str] = []
    if getattr(response, "tool_calls", None):
        # Execute each requested tool and feed its structured JSON result
        # back so the LLM can produce the final natural-language answer
        # with the tool's data embedded (never its own memory).
        messages.append(response)
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args", {}) or {}
            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = handler(**args)
                except Exception as exc:  # tool failure must not crash the chat
                    result = {"error": f"tool {name} failed: {exc}"}
            tool_calls_made.append(name)
            messages.append(
                ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"])
            )

        final_llm = llm.bind(max_tokens=TOOL_ANSWER_MAX_TOKENS)
        response = final_llm.invoke(messages)

    # No tool requested → response already holds the standard RAG answer.
    raw_answer: str = (response.content or "").strip()

    answer, citations = _extract_citations(raw_answer)
    refused = _is_refusal(answer)

    return {
        "answer": answer,
        "citations": citations,
        "refused": refused,
        "model": GENERATION_MODEL,
        "tool_calls": tool_calls_made,
    }


# ---------------------------------------------------------------------------
# Long-term memory summarization (spec.md §12)
# ---------------------------------------------------------------------------

def summarize_history(history: list[dict]) -> str:
    """Summarize a conversation into a short fact list for long-term memory.

    Uses a short, cheap prompt with a small token cap so repeated
    summarization stays inexpensive.
    """
    transcript = "\n".join(
        f"{turn.get('role', '')}: {turn.get('content', '')}"
        for turn in history
        if turn.get("role") in ("user", "assistant")
    )
    prompt = (
        "Summarize this college-chatbot conversation into a short fact list "
        "about the user's interests (e.g. 'interested in CSE admissions, "
        "asked about hostel fees twice'). Max 3 bullet points. Do not include "
        "phone numbers, emails, or other personal identifiers.\n\n"
        f"{transcript}"
    )
    llm = _get_llm().bind(max_tokens=SUMMARY_MAX_TOKENS)
    response = llm.invoke([HumanMessage(content=prompt)])
    return (response.content or "").strip()
