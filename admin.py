"""
admin.py - password-protected Gradio admin panel for the filings database.

Run:  python admin.py      ->  http://127.0.0.1:7861   (login = ADMIN_USER / ADMIN_PASSWORD from .env)

Tabs: Library | Fetch from SEC | Add 10-K PDF | Inspect / Delete / Re-embed
Everything is written to the same permanent SQLite file that app.py reads.
"""
import os
import sys

import gradio as gr

import model
import db_viewer

TICKERS = list(model.COMPANIES)
HEADERS = ["Ticker", "Form", "FY", "Period end", "Filed", "Source", "Sections", "Chunks", "Accession"]


def library_rows():
    return [[f["ticker"], f["form_type"], f["fiscal_year"], f["period_end"], f["filing_date"], f["source"],
             f["n_sections"], f["n_chunks"], f["accession_no"]] for f in model.list_filings()]


def stats_text():
    s = model.db_stats()
    return (f"**{s['filings']}** filings · **{s['sections']}** sections · **{s['chunks']}** chunks · "
            f"{s['db_size_mb']} MB  \n`{s['db_file']}`")


def filing_choices():
    return [(f"{f['ticker']} {f['form_type']} FY{f['fiscal_year']} · filed {f['filing_date']} · {f['source']}",
             f["accession_no"]) for f in model.list_filings()]


def refresh_all():
    return library_rows(), stats_text(), gr.update(choices=filing_choices())


# ── Fetch from SEC ─────────────────────────────────────────────────
def do_fetch(ticker, forms, limit, force_reingest):
    if not forms:
        return "Pick at least one form type."
    try:
        msgs = model.ensure_filings(ticker, list(forms), limit=int(limit) if limit else None,
                                    force_check=True, force_reingest=bool(force_reingest))
    except Exception as e:
        return f"❌ {e}"
    return "\n".join(msgs) or "Nothing to do."


# ── PDFs ───────────────────────────────────────────────────────────
def do_pdf(file, ticker, fiscal_year, period_end, force):
    if file is None:
        return "Upload a PDF first."
    path = file if isinstance(file, str) else getattr(file, "name", str(file))
    try:
        return model.ingest_pdf_10k(path, ticker, int(fiscal_year), period_end.strip(), force=bool(force))
    except Exception as e:
        return f"❌ {e}"


def do_backfill():
    return "\n".join(model.backfill_known_pdfs())


# ── Inspect ────────────────────────────────────────────────────────
def show_filing(acc):
    if not acc:
        return "", gr.update(choices=[], value=None), ""
    f = next((x for x in model.list_filings() if x["accession_no"] == acc), None)
    if not f:
        return "Filing not found.", gr.update(choices=[], value=None), ""
    secs = model.get_sections(acc)
    info = (f"**{f['ticker']} {f['form_type']}** FY{f['fiscal_year']} · period end {f['period_end']} · "
            f"filed {f['filing_date']}  \nsource: {f['source']} {f['source_file'] or ''} · "
            f"{f['n_chunks']} chunks · embed model `{f['embed_model']}`  \n`{acc}`")
    names = list(secs)
    return info, gr.update(choices=names, value=names[0] if names else None), (secs[names[0]][:20000] if names else "")


def show_section(acc, name):
    if not acc or not name:
        return ""
    text = model.get_sections(acc).get(name, "")
    return text[:20000] + (f"\n\n… [{len(text):,} chars total, first 20,000 shown]" if len(text) > 20000 else "")


def do_delete(acc, confirm):
    if not acc:
        return "Select a filing."
    if not confirm:
        return "Tick the confirmation box to delete."
    return "🗑️ Deleted." if model.delete_filing(acc) else "Filing not found."


def do_reembed(acc):
    if not acc:
        return "Select a filing."
    return f"Re-embedded {model.reembed_filing(acc)} chunks."


def do_reembed_stale():
    return "\n".join(model.reembed_stale()) or "All vectors already match the current EMBED_MODEL."


def build_admin() -> gr.Blocks:
    with gr.Blocks(title="Financial Assistant - Admin") as demo:
        gr.Markdown("# 🛠️ Admin - filings database")
        stats = gr.Markdown(stats_text())

        with gr.Tab("📚 Library"):
            table = gr.Dataframe(headers=HEADERS, value=library_rows, interactive=False, wrap=True)
            refresh = gr.Button("Refresh")

        with gr.Tab("⬇️ Fetch from SEC"):
            gr.Markdown("Downloads filings from EDGAR, splits them into Items, embeds them and saves "
                        "everything to the database. Already-stored filings are skipped.")
            with gr.Row():
                t = gr.Dropdown(TICKERS, value=TICKERS[0], label="Company")
                forms = gr.CheckboxGroup(["10-K", "10-Q", "8-K"], value=["10-K", "10-Q"], label="Forms")
                limit = gr.Number(value=2, precision=0, label="Newest N per form")
                force = gr.Checkbox(label="Re-ingest even if stored")
            go = gr.Button("Fetch & store", variant="primary")
            fetch_out = gr.Textbox(label="Result", lines=8)

        with gr.Tab("📄 Add 10-K PDF"):
            gr.Markdown(f"Upload an annual report PDF, or copy `Nvidia.pdf` / `Tesla.pdf` into `{model.PDF_DIR}` "
                        "and use the backfill button.")
            with gr.Row():
                pdf = gr.File(label="10-K PDF", file_types=[".pdf"], type="filepath")
                with gr.Column():
                    pt = gr.Dropdown(TICKERS, value=TICKERS[0], label="Company")
                    fy = gr.Number(value=2026, precision=0, label="Fiscal year")
                    pe = gr.Textbox(value="2026-01-25", label="Period end (YYYY-MM-DD)")
                    pforce = gr.Checkbox(label="Replace if this fiscal year is already stored")
            pgo = gr.Button("Ingest PDF", variant="primary")
            backfill = gr.Button("Backfill known PDFs (Nvidia FY2026 + Tesla FY2025)")
            pdf_out = gr.Textbox(label="Result", lines=6)

        with gr.Tab("🔎 Inspect / manage"):
            pick = gr.Dropdown(choices=filing_choices(), label="Filing")
            info = gr.Markdown()
            sec_pick = gr.Dropdown(choices=[], label="Section")
            sec_text = gr.Textbox(label="Section text", lines=16, max_lines=24)
            with gr.Row():
                re_btn = gr.Button("Re-embed this filing")
                re_all = gr.Button("Re-embed all stale filings")
                confirm = gr.Checkbox(label="I want to delete this filing")
                del_btn = gr.Button("Delete filing", variant="stop")
            manage_out = gr.Textbox(label="Result")
            
        db_viewer.build_db_tab()
        sync = [table, stats, pick]
        refresh.click(refresh_all, None, sync)
        go.click(do_fetch, [t, forms, limit, force], fetch_out).then(refresh_all, None, sync)
        pgo.click(do_pdf, [pdf, pt, fy, pe, pforce], pdf_out).then(refresh_all, None, sync)
        backfill.click(do_backfill, None, pdf_out).then(refresh_all, None, sync)
        pick.change(show_filing, pick, [info, sec_pick, sec_text])
        sec_pick.input(show_section, [pick, sec_pick], sec_text)
        re_btn.click(do_reembed, pick, manage_out)
        re_all.click(do_reembed_stale, None, manage_out)
        del_btn.click(do_delete, [pick, confirm], manage_out).then(refresh_all, None, sync)
    return demo


if __name__ == "__main__":
    password = model._env("ADMIN_PASSWORD")
    if not password:
        sys.exit("Set ADMIN_PASSWORD in .env before starting the admin panel.")
    build_admin().queue().launch(server_name=os.getenv("ADMIN_HOST", "127.0.0.1"),
                                 server_port=int(os.getenv("ADMIN_PORT", "7861")),
                                 auth=(os.getenv("ADMIN_USER", "admin"), password))
