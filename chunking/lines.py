"""Line splitting shared by the chunkers, so every chunker numbers lines the same way."""
from __future__ import annotations


def split_lines(text: str) -> list[str]:
    """Split on "\\n" only, keeping line endings, so "".join(lines) == text and numbering matches editors.

    (str.splitlines also splits on form feeds, \\x1c, \\u2028 etc., which would shift line numbers.)
    A "\\r\\n" ending stays on its line; a trailing newline does not start an extra empty line.
    """
    lines = text.split("\n")
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out
