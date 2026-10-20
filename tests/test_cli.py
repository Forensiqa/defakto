# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from typer.testing import CliRunner

from defakto import __version__
from defakto.cli import app

runner = CliRunner()


class TestCLIVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_version_flag_short(self):
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.stdout
