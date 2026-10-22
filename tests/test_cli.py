# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

import json

from typer.testing import CliRunner

from defakto import __version__
from defakto.cli import app

runner = CliRunner()

TEXT = "This is a short test sentence. It has two sentences in total."


class TestCLIVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_version_flag_short(self):
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestCLIAnalyze:
    def test_text_option(self, make_artifact):
        result = runner.invoke(app, ["analyze", "--text", TEXT, "-a", str(make_artifact())])
        assert result.exit_code == 0, result.output
        assert "forensiqa/test-artifact" in result.stdout

    def test_json_output(self, make_artifact):
        args = ["analyze", "--text", TEXT, "-a", str(make_artifact()), "--json"]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        (data,) = json.loads(result.stdout)
        assert data["artifact_id"] == "forensiqa/test-artifact"
        assert data["num_steps"] == len(TEXT.split())

    def test_reads_standard_input(self, make_artifact):
        args = ["analyze", "-", "-a", str(make_artifact()), "--json"]
        result = runner.invoke(app, args, input=TEXT)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)[0]["num_steps"] == len(TEXT.split())

    def test_reads_a_file(self, make_artifact, tmp_path):
        path = tmp_path / "statement.txt"
        path.write_text(TEXT)
        result = runner.invoke(app, ["analyze", str(path), "-a", str(make_artifact()), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)[0]["source"] == str(path)

    def test_needs_something_to_analyze(self, make_artifact):
        result = runner.invoke(app, ["analyze", "-a", str(make_artifact())])
        assert result.exit_code == 1
        assert "Give a file to analyze" in result.stderr

    def test_needs_artifacts(self):
        result = runner.invoke(app, ["analyze", "--text", TEXT])
        assert result.exit_code == 1
        assert "--artifact" in result.stderr

    def test_reports_errors_without_a_traceback(self, tmp_path):
        result = runner.invoke(app, ["analyze", "--text", TEXT, "-a", str(tmp_path / "nope")])
        assert result.exit_code == 1
        assert "No artifact found" in result.stderr
