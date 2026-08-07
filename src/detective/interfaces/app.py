"""Gradio web UI: chat and report on the left, live evidence on the right.

Clicking a citation scrolls its source passage into view and lights up the exact sentence
the claim rests on. The wiring is one delegated click handler over ``a.cite`` — no
per-render JavaScript and no round trip to the server, because every highlight target is
already in the DOM by the time the report renders.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import gradio as gr

from detective.core.config import Settings
from detective.core.models import Investigation
from detective.interfaces.render import (
    render_panels,
    report_markdown,
    retrieval_trace_markdown,
)
from detective.investigation.pipeline import Investigator
from detective.storage.s3 import upload_report

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gradio.blocks import Blocks

CSS = """
.evidence-pane { max-height: 70vh; overflow-y: auto; padding-right: .5rem; }
.report h3 { margin: 1rem 0 .35rem; font-size: .8rem; text-transform: uppercase;
             letter-spacing: .08em; opacity: .6; }
.report li { margin-bottom: .45rem; line-height: 1.6; }
a.cite { display: inline-block; margin: 0 .15rem; padding: 0 .4rem; border-radius: .7rem;
         background: var(--color-accent-soft); color: var(--body-text-color);
         font-size: .72rem; font-weight: 600; text-decoration: none; cursor: pointer;
         vertical-align: .1rem; white-space: nowrap; }
a.cite:hover { filter: brightness(1.15); }
a.cite.unverified { background: #fde68a; color: #7c2d12; }
.evidence { border: 1px solid var(--border-color-primary); border-radius: .6rem;
            padding: .7rem .8rem; margin-bottom: .6rem; transition: box-shadow .2s ease; }
.evidence.flash { box-shadow: 0 0 0 2px var(--color-accent); }
.evidence-head { display: flex; gap: .5rem; align-items: center; margin-bottom: .4rem; }
.evidence-head .label { font-weight: 700; }
.evidence-head .source { font-family: ui-monospace, monospace; font-size: .8rem; opacity: .75; }
.evidence-head .score { margin-left: auto; font-size: .72rem; padding: .05rem .45rem;
                        border-radius: .5rem; background: var(--color-accent-soft); }
.passage { line-height: 1.6; margin: 0 0 .4rem; }
.line { border-radius: .2rem; padding: 0 .1rem; transition: background .25s ease; }
.line.cited { background: rgba(253, 224, 71, .28); }
.line.active { background: #fde047; box-shadow: 0 0 0 2px #fde047; }
.muted { opacity: .65; font-size: .8rem; }
"""

JS = """
() => {
  document.addEventListener('click', (event) => {
    const cite = event.target.closest('a.cite');
    if (!cite) return;
    event.preventDefault();
    const card = document.getElementById(cite.dataset.anchor);
    if (!card) return;
    document.querySelectorAll('.line.active').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.evidence.flash').forEach(n => n.classList.remove('flash'));
    card.classList.add('flash');
    const numbers = (cite.dataset.lines || '').split(',').filter(Boolean);
    const label = cite.dataset.label;
    let first = null;
    numbers.forEach(n => {
      const line = document.getElementById(`line-${label}-${n}`);
      if (line) { line.classList.add('active'); first = first || line; }
    });
    (first || card).scrollIntoView({ behavior: 'smooth', block: 'center' });
    setTimeout(() => card.classList.remove('flash'), 1800);
  });
}
"""

EMPTY_EVIDENCE = "<p class='muted'>Evidence will appear here once you ask a question.</p>"

#: Short text labels, so the shape of the loop is legible at a glance while it runs.
STEP_LABELS = {
    "plan": "PLAN",
    "search": "SEARCH",
    "verdict": "VERDICT",
    "synthesis": "WRITE",
    "done": "DONE",
}

#: Deliberately spans the corpus: two questions the archive answers well, one whose
#: answer is split across files, and one it should decline to answer.
EXAMPLE_QUESTIONS = [
    "How did the hacker launder the stolen funds?",
    "How did the attacker first gain access?",
    "What techniques were used to obscure the trail?",
    "Did the attacker try to extort the exchange afterwards?",
    "What was the total value of insider trades?",
]


def _progress_markdown(investigation: Investigation) -> str:
    """Render the steps taken so far, with the newest one still in progress."""
    lines: list[str] = []
    for index, step in enumerate(investigation.steps):
        label = STEP_LABELS.get(step.kind, step.kind.upper())
        running = index == len(investigation.steps) - 1 and investigation.report is None
        lines.append(f"`{label}` **{step.headline}**{' …' if running else ''}")
        lines += [f"&nbsp;&nbsp;&nbsp;&nbsp;<sub>{detail}</sub>" for detail in step.detail]
    return "\n\n".join(lines)


INTRO = (
    "## Case Archive Investigator\n"
    "Ask a question about the indexed archive. Every claim in the report is footnoted — "
    "**click a citation to highlight the exact sentence it came from.**"
)


def build_ui(investigator: Investigator, settings: Settings) -> Blocks:
    """Assemble the Blocks app around an already-indexed corpus."""
    state: dict[str, Investigation] = {}

    def respond(
        question: str, history: list[dict[str, str]], mode: str
    ) -> Iterator[tuple[list[dict[str, str]], str, str, str, str]]:
        """Drive the pipeline, re-rendering after every step.

        The investigation object is mutated in place by the generator, so each yield can
        re-render the evidence panel from whatever has been admitted so far. The user
        watches the agent search, accept and reject rather than a spinner.
        """
        question = question.strip()
        if not question:
            yield history, EMPTY_EVIDENCE, "", "", ""
            return

        turns = [*history, {"role": "user", "content": question}]
        investigation = Investigation(question=question, mode=mode)

        for _ in investigator.investigate_stream(investigation):
            _, evidence_html = render_panels(investigation)
            yield (
                [*turns, {"role": "assistant", "content": _progress_markdown(investigation)}],
                evidence_html or EMPTY_EVIDENCE,
                "",
                retrieval_trace_markdown(investigation),
                "",
            )

        state["current"] = investigation
        report_html, evidence_html = render_panels(investigation)
        summary = investigation.report.summary if investigation.report else "No report generated."
        sources = ", ".join(investigation.sources) or "none"
        turns.append(
            {
                "role": "assistant",
                "content": f"{summary}\n\n**Sources:** {sources}\n\n"
                f"<details><summary>How this was found</summary>\n\n"
                f"{_progress_markdown(investigation)}\n\n</details>",
            }
        )
        yield turns, evidence_html, report_html, retrieval_trace_markdown(investigation), ""

    def archive() -> str:
        investigation = state.get("current")
        if investigation is None:
            return "Ask a question first."
        if not settings.s3_configured:
            return "S3 is not configured — set the bucket and credentials in `.env`."
        uri = upload_report(report_markdown(investigation), investigation.question, settings)
        return f"Uploaded to `{uri}`" if uri else "Upload failed."

    with gr.Blocks(title="Case Archive Investigator") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=5):
                chat = gr.Chatbot(height=320, label="Investigation")
                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="e.g. How were the stolen funds moved?",
                        show_label=False,
                        scale=6,
                        autofocus=True,
                    )
                    ask_button = gr.Button("Ask", variant="primary", scale=1)
                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUESTIONS],
                    inputs=[question_box],
                    label="Example questions (the last one the archive cannot answer)",
                )
                mode_picker = gr.Radio(
                    choices=[("Multi-step (agentic)", "agentic"), ("Single-step", "single")],
                    value="agentic",
                    label="Retrieval mode",
                    info=(
                        "Agentic plans a query, rules on what comes back, and keeps searching "
                        "until the evidence holds up. Single-step does one hybrid search."
                    ),
                )
                report_panel = gr.HTML(label="Report")
            with gr.Column(scale=4):
                gr.Markdown("### Evidence")
                evidence_panel = gr.HTML(value=EMPTY_EVIDENCE, elem_classes=["evidence-pane"])
                with gr.Accordion("Retrieval trace", open=False):
                    trace_panel = gr.Markdown()
                upload_button = gr.Button(
                    "Archive report to S3", interactive=settings.s3_configured
                )
                upload_status = gr.Markdown()

        outputs = [chat, evidence_panel, report_panel, trace_panel, question_box]
        submit: dict[str, Any] = {
            "fn": respond,
            "inputs": [question_box, chat, mode_picker],
            "outputs": outputs,
        }
        question_box.submit(**submit)
        ask_button.click(**submit)
        upload_button.click(fn=archive, outputs=upload_status)

    return cast("Blocks", demo)


def launch(investigator: Investigator, settings: Settings, *, host: str, port: int) -> None:
    """Serve the UI.

    Styling and the citation handler are supplied at launch, which is where Gradio 6
    expects them.
    """
    build_ui(investigator, settings).launch(server_name=host, server_port=port, css=CSS, js=JS)
