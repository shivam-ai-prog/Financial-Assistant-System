"""
app.py - Gradio chat UI for the Financial Assistant.

Run:  python app.py        ->  http://127.0.0.1:7860
Data is read from (and new SEC filings are saved to) the permanent SQLite DB used by admin.py.
"""
import json
import traceback
import os
import uuid

import gradio as gr

import model

COMPANY_CHOICES = ["Auto-detect"] + [info["name"] for info in model.COMPANIES.values()]

EXAMPLES = [
    "What was NVIDIA's latest quarterly revenue and how did Data Center perform?",
    "Give a full financial overview of Tesla's latest reported quarter from the 10-Q.",
    "Discuss NVIDIA's gross margin trend and operating expenses from the latest 10-K.",
    "Analyze Tesla's Energy Generation and Storage segment.",
    "Compare NVIDIA and Tesla operating income in the latest 10-K.",
    "Plot NVIDIA and Tesla revenue trend over the last 5 years",
    "Show a margin comparison chart for Tesla",
    "Chart NVIDIA revenue growth year over year",
]


def _filings_table() -> list[list]:
    return [[f["ticker"], f["form_type"], f["fiscal_year"], f["period_end"], f["filing_date"],
             f["source"], f["n_chunks"]] for f in model.list_filings()]


def respond(message, history, company, session_id):
    history = list(history or [])
    if not message or not message.strip():
        yield history, "", "{}"
        return
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⏳ Checking stored filings, retrieving and analysing…"})
    yield history, "", "{}"

    figs = []
    hint = None if company.startswith("Auto") else company
    try:
        out = model.financial_assistant(message, hint, session_id)
        answer, figs = out["answer"], out.get("charts") or []
        details = json.dumps({"router": out["decision"], "sec_updates": out["ingest"],
                              "sources": out["sources"]}, indent=2, default=str)
    except Exception as e:
        traceback.print_exc()  
        answer, details = f"⚠️ Something went wrong: `{e}`", "{}"
    history.pop()  # remove the placeholder
    for fig in figs:
        history.append({"role": "assistant", "content": gr.Plot(fig)})
    history.append({"role": "assistant", "content": answer})
    yield history, "", details


def _chatbot():
    try:
        return gr.Chatbot(type="messages", height=560, label="Financial Assistant")
    except TypeError:  # newer Gradio: messages format is the only one
        return gr.Chatbot(height=560, label="Financial Assistant")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Financial Assistant") as demo:
        gr.Markdown("# 📈 Financial Assistant\nAnswers from SEC filings (10-K / 10-Q / 8-K) stored in the local "
                    "database. New filings are fetched from SEC EDGAR automatically and saved permanently.")
        session_id = gr.State(lambda: uuid.uuid4().hex)
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = _chatbot()
                with gr.Row():
                    box = gr.Textbox(placeholder="Ask about revenue, margins, risks, strategy…", scale=5,
                                     show_label=False, autofocus=True)
                    company = gr.Dropdown(COMPANY_CHOICES, value="Auto-detect", label="Company", scale=1)
                    send = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(EXAMPLES, inputs=box, label="Try one")
                clear = gr.Button("New conversation")
            with gr.Column(scale=1):
                with gr.Accordion("How this answer was built", open=False):
                    details = gr.Code(language="json", label="router / SEC updates / sources")
                with gr.Accordion("Filings in the database", open=True):
                    table = gr.Dataframe(headers=["Ticker", "Form", "FY", "Period end", "Filed", "Source", "Chunks"],
                                         value=_filings_table, interactive=False, wrap=True)
                    refresh = gr.Button("Refresh list")

        inputs, outputs = [box, chatbot, company, session_id], [chatbot, box, details]
        # inputs order == respond(message, history, company, session_id); respond is a generator (streams)
        box.submit(respond, inputs, outputs).then(_filings_table, None, table)
        send.click(respond, inputs, outputs).then(_filings_table, None, table)
        clear.click(lambda: ([], "{}", uuid.uuid4().hex), None, [chatbot, details, session_id])
        refresh.click(_filings_table, None, table)
    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name=os.getenv("APP_HOST", "127.0.0.1"),
                              server_port=int(os.getenv("APP_PORT", "7860")))