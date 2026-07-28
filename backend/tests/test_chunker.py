"""Tests for AST chunking (`chunker.chunk_file`)."""

from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_file
from app.ingestion.models import Chunk

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    # read_text uses universal newlines, so `source` is always LF regardless of
    # how the fixture is stored on disk -- keeps line math platform-stable.
    return (FIXTURES / name).read_text(encoding="utf-8")


def _by_qname(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.qualified_name: c for c in chunks if c.chunk_type != "module"}


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #


def test_python_chunk_count() -> None:
    source = _read("sample.py")
    chunks = chunk_file("sample.py", source)
    # 5 definitions (2 functions incl. nested, 1 class, 2 methods) + 1 module.
    assert len(chunks) == 6
    assert sum(1 for c in chunks if c.chunk_type == "module") == 1


def test_python_definition_metadata() -> None:
    source = _read("sample.py")
    defs = _by_qname(chunk_file("sample.py", source))

    expected = {
        "top_level_function": ("function", None),
        "top_level_function.helper": ("function", "top_level_function"),
        "UserAuth": ("class", None),
        "UserAuth.__init__": ("method", "UserAuth"),
        "UserAuth.validate_token": ("method", "UserAuth"),
    }
    assert set(defs) == set(expected)
    for qname, (chunk_type, parent) in expected.items():
        assert defs[qname].chunk_type == chunk_type, qname
        assert defs[qname].parent_name == parent, qname
        assert defs[qname].language == "python"


def test_python_nested_function_is_independently_chunked() -> None:
    defs = _by_qname(chunk_file("sample.py", _read("sample.py")))
    helper = defs["top_level_function.helper"]
    assert helper.name == "helper"
    assert helper.parent_name == "top_level_function"
    assert helper.chunk_type == "function"


def test_python_line_bounds_slice_back_to_content() -> None:
    source = _read("sample.py")
    for chunk in chunk_file("sample.py", source):
        if chunk.chunk_type == "module":
            continue
        # The declared [start_line, end_line] must reproduce content exactly.
        assert chunk.slice_from(source) == chunk.content, chunk.qualified_name
        first_line = source.splitlines()[chunk.start_line - 1]
        assert chunk.name in first_line


def test_python_docstrings() -> None:
    defs = _by_qname(chunk_file("sample.py", _read("sample.py")))
    assert defs["top_level_function"].docstring == "Return x doubled."
    assert defs["top_level_function.helper"].docstring == "Nested helper."
    assert defs["UserAuth"].docstring == "Handles auth."
    assert defs["UserAuth.validate_token"].docstring == "Check a token."
    assert defs["UserAuth.__init__"].docstring is None


def test_python_module_chunk() -> None:
    chunks = chunk_file("sample.py", _read("sample.py"))
    module = next(c for c in chunks if c.chunk_type == "module")
    assert module.name == "sample"
    assert module.parent_name is None
    assert module.docstring == "Module docstring for sample."
    # Top-level code is preserved, not dropped.
    assert "import os" in module.content
    assert "CONSTANT = 42" in module.content
    assert 'if __name__ == "__main__":' in module.content
    # Function/class bodies live in their own chunks, not the module chunk.
    assert "def top_level_function" not in module.content


# --------------------------------------------------------------------------- #
# TypeScript
# --------------------------------------------------------------------------- #


def test_typescript_chunk_count() -> None:
    chunks = chunk_file("sample.ts", _read("sample.ts"))
    # add, inner, UserAuth, constructor, validateToken, multiply + module.
    assert len(chunks) == 7


def test_typescript_definition_metadata() -> None:
    defs = _by_qname(chunk_file("sample.ts", _read("sample.ts")))
    expected = {
        "add": ("function", None),
        "add.inner": ("function", "add"),
        "UserAuth": ("class", None),
        "UserAuth.constructor": ("method", "UserAuth"),
        "UserAuth.validateToken": ("method", "UserAuth"),
        "multiply": ("function", None),
    }
    assert set(defs) == set(expected)
    for qname, (chunk_type, parent) in expected.items():
        assert defs[qname].chunk_type == chunk_type, qname
        assert defs[qname].parent_name == parent, qname
        assert defs[qname].language == "typescript"


def test_typescript_nested_and_method_chunking() -> None:
    defs = _by_qname(chunk_file("sample.ts", _read("sample.ts")))
    assert defs["add.inner"].chunk_type == "function"
    assert defs["add.inner"].parent_name == "add"
    assert defs["UserAuth.validateToken"].chunk_type == "method"
    assert defs["multiply"].chunk_type == "function"  # arrow function


def test_typescript_line_bounds_slice_back_to_content() -> None:
    source = _read("sample.ts")
    for chunk in chunk_file("sample.ts", source):
        if chunk.chunk_type == "module":
            continue
        assert chunk.slice_from(source) == chunk.content, chunk.qualified_name


def test_typescript_jsdoc_and_line_comment_docstrings() -> None:
    defs = _by_qname(chunk_file("sample.ts", _read("sample.ts")))
    assert defs["add"].docstring == "Adds two numbers together."
    assert defs["UserAuth"].docstring == "A simple user authentication class"
    assert defs["UserAuth.validateToken"].docstring == "Validate a candidate token."
    assert defs["add.inner"].docstring is None
    assert defs["UserAuth.constructor"].docstring is None


def test_typescript_module_chunk_has_import() -> None:
    chunks = chunk_file("sample.ts", _read("sample.ts"))
    module = next(c for c in chunks if c.chunk_type == "module")
    assert "import" in module.content


# --------------------------------------------------------------------------- #
# Robustness / fallback
# --------------------------------------------------------------------------- #


def test_syntax_error_falls_back_to_single_module_chunk() -> None:
    source = _read("broken.py")
    chunks = chunk_file("broken.py", source)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module"
    assert chunks[0].content == source
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == source.count("\n") + 1
    assert chunks[0].language == "python"


def test_unsupported_extension_falls_back() -> None:
    chunks = chunk_file("notes.md", "# just markdown\n")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module"


def test_empty_file_produces_no_chunks() -> None:
    assert chunk_file("empty.py", "") == []
    assert chunk_file("empty.py", "   \n  \n") == []


def test_javascript_arrow_and_class() -> None:
    source = "function f(){ return 1; }\nclass A { m(){ return 2; } }\nconst h = () => 3;\n"
    defs = _by_qname(chunk_file("thing.js", source))
    assert defs["f"].chunk_type == "function"
    assert defs["f"].language == "javascript"
    assert defs["A"].chunk_type == "class"
    assert defs["A.m"].chunk_type == "method"
    assert defs["h"].chunk_type == "function"


def test_repo_metadata_propagates() -> None:
    chunks = chunk_file(
        "sample.py", _read("sample.py"), repo_url="https://x/y", commit_sha="abc123"
    )
    assert all(c.repo_url == "https://x/y" for c in chunks)
    assert all(c.commit_sha == "abc123" for c in chunks)
