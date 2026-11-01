"""
db_viewer.py - a live, READ-ONLY view of the SQLite database, used as a tab inside admin.py.
Opens data/filings_index.db straight from disk every time, so nothing is ever uploaded or copied.
"""
import re
import sqlite3

import gradio as gr
import pandas as pd

import model

_SELECT_ONLY = re.compile(r"^\s*(select|with|pragma\s+table_info)\b", re.I)


def _ro():
    """Read-only connection: the viewer can never change or lock your data."""
    conn = sqlite3.connect(f"file:{model.DB_PATH.as_posix()}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def table_names() -> list[str]:
    try:
        with _ro() as c:
            return [r[0] for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    except sqlite3.Error:
        return []


def _frame(cursor) -> pd.DataFrame:
    cols = [d[0] for d in cursor.description]
    rows = []
    for r in cursor.fetchall():
        rows.append([f"<{len(v)} bytes>" if isinstance(v, (bytes, bytearray))  # embeddings: show size, not bytes
                     else (v[:300] + " ..." if isinstance(v, str) and len(v) > 300 else v) for v in r])
    return pd.DataFrame(rows, columns=cols)


def overview() -> pd.DataFrame:
    out = []
    with _ro() as c:
        for t in table_names():
            out.append({"table": t, "rows": c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]})
    return pd.DataFrame(out)


def show_table(name: str, limit: int, newest_first: bool):
    if not name:
        return pd.DataFrame(), "Pick a table."
    try:
        with _ro() as c:
            total = c.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            order = " ORDER BY rowid DESC" if newest_first else ""
            cur = c.execute(f'SELECT * FROM "{name}"{order} LIMIT ?', (int(limit or 100),))
            df = _frame(cur)
        return df, f"**{name}** - {total} rows in total (showing {len(df)}). Database: `{model.DB_PATH}`"
    except sqlite3.Error as e:
        return pd.DataFrame(), f"⚠️ {e}"


def run_sql(sql: str):
    if not _SELECT_ONLY.match(sql or ""):
        return pd.DataFrame(), "Only read-only queries are allowed (SELECT ...)."
    try:
        with _ro() as c:
            df = _frame(c.execute(sql))
        return df, f"{len(df)} rows."
    except sqlite3.Error as e:
        return pd.DataFrame(), f"⚠️ {e}"


def build_db_tab():
    """Call inside `with gr.Blocks(...)`: adds a '🗄️ Database' tab."""
    with gr.Tab("🗄️ Database"):
        gr.Markdown("Live, read-only view of the real database file. Click **Refresh** (or tick auto-refresh) "
                    "after an ingest or delete - no upload needed.")
        with gr.Row():
            tbl = gr.Dropdown(choices=table_names(), value="filings" if "filings" in table_names() else None,
                              label="Table", scale=2)
            limit = gr.Number(value=100, precision=0, label="Rows to show", scale=1)
            newest = gr.Checkbox(value=True, label="Newest first", scale=1)
            refresh = gr.Button("🔄 Refresh", variant="primary", scale=1)
            auto = gr.Checkbox(label="Auto-refresh every 5 s", scale=1)
        info = gr.Markdown(value=lambda: show_table("filings", 100, True)[1])
        grid = gr.Dataframe(value=lambda: show_table("filings", 100, True)[0], interactive=False, wrap=True,
                            max_height=480)
        with gr.Accordion("Table sizes", open=False):
            sizes = gr.Dataframe(value=overview, interactive=False)
        with gr.Accordion("Vector index (search layer)", open=False):
            vinfo = gr.Markdown()
            vgrid = gr.Dataframe(interactive=False, wrap=True)
            with gr.Row():
                vcheck = gr.Button("Check index")
                vsync = gr.Button("Repair / sync index now")

            def _vec(repair=False):
                import vectorstore
                try:
                    if repair:
                        vectorstore.sync(force=True)
                    st = vectorstore.status()
                    msg = (f"**Backend:** {st.get('backend')} - {'OK' if st.get('ok') else 'PROBLEM'}  \n"
                           f"{st.get('detail')}  \nSQLite chunks: {st.get('sqlite_chunks')}")
                    return pd.DataFrame(vectorstore.per_filing_report()), msg
                except Exception as e:
                    return pd.DataFrame(), f"Vector layer error: {e}"
            vcheck.click(lambda: _vec(False), None, [vgrid, vinfo])
            vsync.click(lambda: _vec(True), None, [vgrid, vinfo])
        with gr.Accordion("Run your own SELECT query", open=False):
            sql = gr.Textbox(value="SELECT ticker, form_type, fiscal_year, period_end, n_chunks FROM filings",
                             lines=3, label="SQL (read-only)")
            run = gr.Button("Run")
            sql_info = gr.Markdown()
            sql_grid = gr.Dataframe(interactive=False, wrap=True)

        def _reload(name, lim, nf):
            df, msg = show_table(name, lim, nf)
            names = table_names()
            return gr.update(choices=names), df, msg, overview()

        outs = [tbl, grid, info, sizes]
        ins = [tbl, limit, newest]
        refresh.click(_reload, ins, outs)
        tbl.change(_reload, ins, outs)
        limit.change(_reload, ins, outs)
        newest.change(_reload, ins, outs)
        run.click(run_sql, sql, [sql_grid, sql_info])
        try:
            timer = gr.Timer(5, active=False)
            timer.tick(_reload, ins, outs)
            auto.change(lambda on: gr.Timer(active=bool(on)), auto, timer)
        except AttributeError:  # older Gradio without gr.Timer: use the Refresh button
            auto.visible = False
        return tbl, ins, outs, _reload