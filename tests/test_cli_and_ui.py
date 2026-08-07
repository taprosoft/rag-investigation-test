from __future__ import annotations

from pathlib import Path

import pytest

from conftest import json_response, make_investigator
from detective.__main__ import build_parser, build_providers, main
from detective.core.config import Settings
from detective.interfaces.app import JS, build_ui


class TestProviderWiring:
    def test_offline_needs_no_credentials(self) -> None:
        providers = build_providers(Settings(), offline=True)

        assert providers.embedder.embed(["x"]).shape[0] == 1

    def test_live_mode_names_the_missing_keys(self) -> None:
        with pytest.raises(SystemExit, match="OPENAI_API_KEY, COHERE_API_KEY"):
            build_providers(Settings(openai_api_key="", cohere_api_key=""), offline=False)


class TestCli:
    def test_ask_prints_a_report(
        self, repo_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkey_corpus = str(repo_root / "eval" / "synthetic")

        exit_code = main(
            [
                "ask",
                "How did the intruder get in?",
                "--corpus",
                monkey_corpus,
                "--offline",
                "--trace",
            ]
        )

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "# Investigation report" in out
        assert "Retrieval trace" in out
        assert "Round 1" in out

    def test_eval_prints_the_comparison_table(
        self, repo_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output = tmp_path / "table.md"

        exit_code = main(
            [
                "eval",
                "--goldens",
                str(repo_root / "eval" / "goldens.json"),
                "--offline",
                "--output",
                str(output),
            ]
        )

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "| hybrid+rerank |" in out
        assert "| hybrid+rerank |" in output.read_text(encoding="utf-8")

    def test_parser_defaults_to_the_agentic_mode(self) -> None:
        args = build_parser().parse_args(["ask", "why?"])

        assert args.mode == "agentic"
        assert args.offline is False


class TestUi:
    def test_builds_with_both_panels_and_the_citation_handler(self, settings: Settings) -> None:
        investigator = make_investigator(settings, [json_response({"summary": "ok"})])

        demo = build_ui(investigator, Settings())

        assert demo is not None
        assert "a.cite" in JS
        assert "scrollIntoView" in JS

    def test_upload_button_is_disabled_without_s3_configuration(self, settings: Settings) -> None:
        investigator = make_investigator(settings, [json_response({})])

        demo = build_ui(investigator, Settings())
        buttons = [
            block
            for block in demo.blocks.values()
            if getattr(block, "value", None) == "Archive report to S3"
        ]

        assert buttons
        assert getattr(buttons[0], "interactive", None) is False
