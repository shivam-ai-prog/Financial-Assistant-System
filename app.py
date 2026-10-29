"""
app.py - Gradio chat UI for the Financial Assistant.

Run:  python app.py        ->  http://127.0.0.1:7860
Data is read from (and new SEC filings are saved to) the permanent SQLite DB used by admin.py.
"""
import json
import logging
import traceback
import os
import uuid

import gradio as gr

import companies
import model

def _company_choices() -> list[str]:
    try:
        return ["Auto-detect"] + companies.choices()
    except Exception:
        return ["Auto-detect"] + [info["name"] for info in model.COMPANIES.values()]


COMPANY_CHOICES = _company_choices()

EXAMPLES = [
    "What was NVIDIA's latest quarterly revenue and how did Data Center perform?",
    "Give a full financial overview of Tesla's latest reported quarter from the 10-Q.",
    "Discuss NVIDIA's gross margin trend and operating expenses from the latest 10-K.",
    "Analyze Tesla's Energy Generation and Storage segment.",
    "Compare NVIDIA and Tesla operating income in the latest 10-K.",
    "Plot NVIDIA and Tesla revenue trend over the last 5 years",
    "What were Apple's main risk factors in its latest 10-K?",
    "Compare Microsoft and Alphabet operating margin as a chart",
    "Show a margin comparison chart for Tesla",
    "Chart NVIDIA revenue growth year over year",
    "Evaluate NVIDIA's performance with key financial metrics, year over year and quarterly",
    "Benchmark NVIDIA, AMD and Intel on profitability, growth and cash flow",
    "Generate a financial report for Tesla",
    "Generate a financial report comparing NVIDIA and AMD",
]


def _filings_table() -> list[list]:
    return [[f["ticker"], f["form_type"], f["fiscal_year"], f["period_end"], f["filing_date"],
             f["source"], f["n_chunks"]] for f in model.list_filings()]


def respond(message, history, company, session_id):
    history = list(history or [])
    if not message or not message.strip():
        yield history, "", None
        return
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⏳ Checking stored filings, retrieving and analysing… (a company seen for the first time is "
                                                "downloaded from SEC EDGAR first - this can take a few minutes)"})
    yield history, "", None

    figs, report_file = [], None
    hint = None if company.startswith("Auto") else company
    try:
        out = model.financial_assistant(message, hint, session_id)
        answer, figs = out["answer"], out.get("charts") or []
        report_file = out.get("report_file")
        model.log.info("router: %s", json.dumps(out["decision"], default=str))
    except Exception as e:  # show the problem instead of crashing the UI
        traceback.print_exc()  # full error with file + line goes to the terminal
        answer = f"⚠️ Something went wrong: `{e}`"
    history.pop()  # remove the placeholder
    for fig in figs:
        history.append({"role": "assistant", "content": gr.Plot(fig)})
    history.append({"role": "assistant", "content": answer})
    yield history, "", report_file


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
                    company = gr.Dropdown(COMPANY_CHOICES, value="Auto-detect", label="Company (or type any)", scale=1,
                                          allow_custom_value=True)
                    send = gr.Button("Send", variant="primary", scale=1)
                report_box = gr.File(label="📄 Download report (.docx) - appears after 'generate a financial report'",
                                     interactive=False)
                gr.Examples(EXAMPLES, inputs=box, label="Try one")
                clear = gr.Button("New conversation")

        inputs, outputs = [box, chatbot, company, session_id], [chatbot, box, report_box]
        # inputs order == respond(message, history, company, session_id); respond is a generator (streams)
        box.submit(respond, inputs, outputs).then(lambda: gr.update(choices=_company_choices()), None, company)
        send.click(respond, inputs, outputs).then(
            lambda: gr.update(choices=_company_choices()), None, company)
        clear.click(lambda: ([], uuid.uuid4().hex, None), None, [chatbot, session_id, report_box])
    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name=os.getenv("APP_HOST", "127.0.0.1"),
                              server_port=int(os.getenv("APP_PORT", "7860")),
                              allowed_paths=[str(model.DATA_DIR)])