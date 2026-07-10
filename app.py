"""
app.py
------
Gradio front-end for the Financial Research Assistant.

All heavy lifting (PDF ingestion, embeddings, vector store, LLM, hybrid
retrieval, reranking, BERTScore) lives in model.py. This file only wires
up the chat UI on top of it.

Run with:
    python app.py
"""

import os
import re
import uuid

import gradio as gr

import model


# ---------------------------------------------------------------------------
# UI helper functions
# ---------------------------------------------------------------------------
def _guess_company_names():
    names = set()
    try:
        for f in model.pdf_files:
            stem = f.stem
            first_token = re.split(r"[\s_\-]+", stem)[0]
            if first_token:
                names.add(first_token)
    except NameError:
        pass
    return sorted(names)


def _strip_sources_block(markdown_text: str) -> str:
    return re.split(r"\n##\s*Sources\b", markdown_text, maxsplit=1)[0].strip()


def _new_session_id():
    return str(uuid.uuid4())


def respond(message, chat_history, company, session_id):
    if not message or not message.strip():
        return chat_history, ""

    company_filter = None if company == "All Companies" else company

    try:
        raw_answer = model.financial_rag(
            query=message,
            company=company_filter,
            session_id=session_id,
        )
    except Exception as e:
        raw_answer = f"❌ Something went wrong while generating the answer:\n\n`{e}`"

    clean_answer = _strip_sources_block(raw_answer)

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": clean_answer},
    ]
    return chat_history, ""


def on_company_change(company, session_id):
    # New company -> fresh conversation thread (own memory + own session id),
    # since financial_rag's memory is keyed on (company, session_id).
    return [], _new_session_id()


def clear_chat(session_id):
    return [], _new_session_id()


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
def build_demo():
    company_choices = ["All Companies"] + _guess_company_names()

    with gr.Blocks(title="Financial Research Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📊 Financial Research Assistant
            Ask questions about the SEC filings loaded into the knowledge base.
            Pick a company to focus the search and keep a separate conversation per company.
            """
        )

        with gr.Row():
            company_dropdown = gr.Dropdown(
                choices=company_choices,
                value=company_choices[0],
                label="Company",
                scale=3,
            )
            clear_btn = gr.Button("🗑️ New conversation", scale=1)

        chatbot = gr.Chatbot(
            type="messages",
            height=520,
            label="Conversation",
            show_copy_button=True,
        )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="e.g. What was the revenue breakdown by segment in the latest 10-K?",
                label="Your question",
                scale=5,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        gr.Examples(
            examples=[
                "What is the total revenue in the latest filing and how does it compare to the previous year?",
                "Summarize the key risk factors mentioned in the filing.",
                "What are the gross margin trends and what's driving them?",
                "Give a comprehensive financial analysis with investor takeaways.",
            ],
            inputs=msg_box,
        )

        session_state = gr.State(_new_session_id())

        send_btn.click(
            respond,
            inputs=[msg_box, chatbot, company_dropdown, session_state],
            outputs=[chatbot, msg_box],
        )
        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, company_dropdown, session_state],
            outputs=[chatbot, msg_box],
        )
        company_dropdown.change(
            on_company_change,
            inputs=[company_dropdown, session_state],
            outputs=[chatbot, session_state],
        )
        clear_btn.click(
            clear_chat,
            inputs=[session_state],
            outputs=[chatbot, session_state],
        )

    return demo


if __name__ == "__main__":
    # Set PDF_DIR / CHROMA_PERSIST_DIR / HF_TOKEN via environment variables,
    # e.g.:
    #   export HF_TOKEN="hf_xxx"
    #   export PDF_DIR="./data/SEC Filings"
    #   export CHROMA_PERSIST_DIR="./financial_db"
    model.initialize()

    demo = build_demo()
    demo.launch(
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
        debug=False,
    )
