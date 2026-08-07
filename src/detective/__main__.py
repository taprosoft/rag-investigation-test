"""Command line entry point: ``ask``, ``serve`` and ``eval``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from detective.core.config import Settings
from detective.evaluation import evaluate_all, format_table, load_goldens
from detective.interfaces.render import report_markdown, retrieval_trace_markdown
from detective.investigation.pipeline import Investigator, Providers
from detective.providers import (
    CohereReranker,
    HashEmbedder,
    LexicalOverlapReranker,
    OpenAIChatModel,
    OpenAIEmbedder,
    ScriptedChatModel,
)
from detective.storage.s3 import upload_report

OFFLINE_HELP = "use deterministic local stand-ins instead of the model APIs"


def build_providers(settings: Settings, *, offline: bool) -> Providers:
    """Wire live providers, or deterministic stand-ins when offline.

    The offline path exists so the retrieval machinery can be demonstrated and evaluated
    without credentials; it is not a quality claim, and the CLI says so.
    """
    if offline:
        return Providers(
            embedder=HashEmbedder(),
            reranker=LexicalOverlapReranker(),
            chat=ScriptedChatModel(['{"sufficient": true, "summary": "offline stub"}']),
        )
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", settings.openai_api_key),
            ("COHERE_API_KEY", settings.cohere_api_key),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"missing credentials: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in, or pass --offline."
        )
    return Providers(
        embedder=OpenAIEmbedder(settings.openai_api_key, settings.embedding_model),
        reranker=CohereReranker(settings.cohere_api_key, settings.rerank_model),
        chat=OpenAIChatModel(settings.openai_api_key, settings.chat_model),
    )


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if getattr(args, "corpus", None):
        settings = settings.model_copy(update={"corpus_dir": Path(args.corpus)})
    return settings


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    investigator = Investigator.build(build_providers(settings, offline=args.offline), settings)
    investigation = investigator.investigate(args.question, mode=args.mode)

    markdown = report_markdown(investigation)
    print(markdown)
    if args.trace:
        print("\n## Retrieval trace\n")
        print(retrieval_trace_markdown(investigation))

    if args.upload:
        uri = upload_report(markdown, investigation.question, settings)
        print(f"\nUploaded: {uri}" if uri else "\nS3 not configured; upload skipped.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from detective.interfaces.app import launch

    settings = _settings(args)
    investigator = Investigator.build(build_providers(settings, offline=args.offline), settings)
    launch(investigator, settings, host=args.host, port=args.port)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    goldens = load_goldens(Path(args.goldens))
    settings = Settings().model_copy(update={"corpus_dir": goldens.corpus_dir})
    investigator = Investigator.build(
        build_providers(settings, offline=args.offline), settings, use_cache=False
    )
    results = evaluate_all(investigator, goldens, top_k=args.top_k, include_pipeline=args.pipeline)

    print(
        f"Corpus: {goldens.corpus_dir} · {len(goldens.queries)} labelled queries · "
        f"top_k={args.top_k} · "
        f"providers={'offline stand-ins' if args.offline else 'live'}\n"
    )
    print(format_table(results))
    if args.output:
        Path(args.output).write_text(format_table(results) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detective", description="Retrieval-augmented investigation over a case archive."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="answer one question and print the report")
    ask.add_argument("question")
    ask.add_argument(
        "--mode", choices=("agentic", "single"), default="agentic", help="retrieval strategy"
    )
    ask.add_argument("--corpus", help="folder of .txt case files (default: data)")
    ask.add_argument("--trace", action="store_true", help="show how the evidence was found")
    ask.add_argument("--upload", action="store_true", help="archive the report to S3")
    ask.add_argument("--offline", action="store_true", help=OFFLINE_HELP)
    ask.set_defaults(handler=cmd_ask)

    serve = subparsers.add_parser("serve", help="launch the web UI")
    serve.add_argument("--corpus", help="folder of .txt case files (default: data)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--offline", action="store_true", help=OFFLINE_HELP)
    serve.set_defaults(handler=cmd_serve)

    evaluate = subparsers.add_parser("eval", help="score retrieval strategies on the golden set")
    evaluate.add_argument("--goldens", default="eval/goldens.json")
    evaluate.add_argument("--top-k", type=int, default=3)
    evaluate.add_argument("--output", help="also write the results table to this path")
    evaluate.add_argument(
        "--pipeline",
        action="store_true",
        help="also score the end-to-end modes (spends LLM calls per query)",
    )
    evaluate.add_argument("--offline", action="store_true", help=OFFLINE_HELP)
    evaluate.set_defaults(handler=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
