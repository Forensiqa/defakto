# Copyright (c) 2026 Forensiqa
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import typer

from defakto import __version__

app = typer.Typer(
    help="Open-source toolkit for detecting AI-generated and unreliable content, from Forensiqa.",
    no_args_is_help=True,
    add_completion=False,
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"defakto {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=_show_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Open-source toolkit for detecting AI-generated and unreliable content, from Forensiqa."""
