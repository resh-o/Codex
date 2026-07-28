"""tree-sitter AST chunking.

Parses Python / TypeScript / JavaScript (incl. TSX/JSX) source into
:class:`~app.ingestion.models.Chunk` objects at function / method / class
boundaries, plus a per-file ``module`` chunk for top-level code.

Design notes
------------
* Package choice: ``tree-sitter`` (the modern >=0.23 binding) with
  ``tree-sitter-language-pack``.  The older ``tree-sitter-languages`` package is
  effectively unmaintained and pins ``tree-sitter<0.22``; ``language-pack`` is
  the actively maintained successor and ships prebuilt wheels for all the
  grammars we need (python, typescript, tsx, javascript).
* Nested functions and methods each become their own independently retrievable
  chunk, with ``parent_name`` / ``qualified_name`` set from the enclosing
  definition chain.
* Robustness first: if a file fails to parse (``root_node.has_error``) or its
  language is unsupported, we emit a single whole-file ``module`` chunk and log
  a warning rather than crashing.  Stage 5 (eval) can then score AST-chunked vs
  fallback-chunked files separately.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Optional

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

from .models import Chunk

logger = logging.getLogger(__name__)

# Extension -> (grammar name for the parser, Chunk.language value).
# TSX and JSX need different grammars but map to the same reported language.
_EXT_CONFIG: dict[str, tuple[str, str]] = {
    ".py": ("python", "python"),
    ".ts": ("typescript", "typescript"),
    ".tsx": ("tsx", "typescript"),
    ".js": ("javascript", "javascript"),
    ".jsx": ("javascript", "javascript"),
}

# Node types that introduce a named definition, per grammar family.
# Value is the "raw kind": one of "class", "function", "method".  Whether a
# "function" becomes a chunk of type function or method depends on its parent
# (a function directly inside a class body is a method) -- see ``_chunk_type``.
_PY_DEFS = {"function_definition": "function", "class_definition": "class"}
_JS_DEFS = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
}
_JS_ARROW_VALUES = {"arrow_function", "function_expression"}


class UnsupportedLanguageError(ValueError):
    """Raised internally when a file extension has no configured grammar."""


def language_for_path(file_path: str) -> Optional[str]:
    """Return the ``Chunk.language`` value for a path, or None if unsupported."""
    cfg = _EXT_CONFIG.get(PurePosixPath(file_path).suffix)
    return cfg[1] if cfg else None


@lru_cache(maxsize=None)
def _parser_for(grammar: str) -> Parser:
    """Return a cached parser for a grammar name (python/typescript/tsx/js)."""
    return get_parser(grammar)  # type: ignore[arg-type]


def chunk_file(
    file_path: str,
    source: str,
    repo_url: str = "",
    commit_sha: str = "",
) -> list[Chunk]:
    """Chunk a single source file into :class:`Chunk` objects.

    ``file_path`` is used both to select the grammar (by extension) and as the
    reported ``Chunk.file_path`` -- pass it relative to the repo root.

    Never raises for parse problems: unsupported or unparseable files degrade to
    a single whole-file ``module`` chunk.
    """
    cfg = _EXT_CONFIG.get(PurePosixPath(file_path).suffix)
    if cfg is None:
        logger.warning("No grammar for %s; emitting fallback module chunk", file_path)
        return _fallback_chunks(file_path, source, "unsupported", repo_url, commit_sha)

    grammar, language = cfg
    source_bytes = source.encode("utf-8")

    try:
        tree = _parser_for(grammar).parse(source_bytes)
    except Exception as exc:  # pragma: no cover - grammar/binding failure
        logger.warning("Parse failed for %s (%s); using fallback", file_path, exc)
        return _fallback_chunks(file_path, source, language, repo_url, commit_sha)

    root = tree.root_node
    if root.has_error:
        logger.warning(
            "Syntax errors in %s; using whole-file fallback module chunk", file_path
        )
        return _fallback_chunks(file_path, source, language, repo_url, commit_sha)

    ctx = _Context(
        file_path=file_path,
        language=language,
        grammar=grammar,
        source=source,
        source_bytes=source_bytes,
        repo_url=repo_url,
        commit_sha=commit_sha,
    )
    chunks: list[Chunk] = []
    _walk(root, ancestors=(), ctx=ctx, out=chunks)

    module_chunk = _module_chunk(root, ctx)
    if module_chunk is not None:
        chunks.append(module_chunk)

    # Stable order: by start line, then by how deep the qualified name is.
    chunks.sort(key=lambda c: (c.start_line, c.qualified_name.count("."), c.end_line))
    return chunks


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


class _Context:
    """Bundle of per-file state threaded through the recursion."""

    __slots__ = (
        "file_path",
        "language",
        "grammar",
        "source_bytes",
        "lines",
        "repo_url",
        "commit_sha",
    )

    def __init__(
        self,
        file_path: str,
        language: str,
        grammar: str,
        source: str,
        source_bytes: bytes,
        repo_url: str,
        commit_sha: str,
    ) -> None:
        self.file_path = file_path
        self.language = language
        self.grammar = grammar
        self.source_bytes = source_bytes
        # Split on "\n" only, matching tree-sitter's row semantics exactly.
        self.lines = source.split("\n")
        self.repo_url = repo_url
        self.commit_sha = commit_sha

    @property
    def is_python(self) -> bool:
        return self.grammar == "python"

    def text(self, node: Node) -> str:
        """Exact source text of a node, by byte range (starts at the node)."""
        return self.source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def line_text(self, start_row: int, end_row: int) -> str:
        """Whole-line text for an inclusive 0-indexed row range.

        Unlike :meth:`text`, this includes leading indentation on the first line
        and trailing content to the end of the last line, so a chunk's content
        equals the exact lines its ``start_line``/``end_line`` claim.
        """
        return "\n".join(self.lines[start_row : end_row + 1])


def _node_name(node: Node) -> Optional[str]:
    """Best-effort definition name via the grammar's ``name`` field."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return name_node.text.decode("utf-8", errors="replace")
    return None


def _detect_definition(node: Node, ctx: _Context) -> Optional[tuple[str, str]]:
    """Return ``(name, raw_kind)`` if ``node`` starts a definition, else None.

    ``raw_kind`` is one of "class" | "function" | "method".
    """
    t = node.type
    if ctx.is_python:
        kind = _PY_DEFS.get(t)
        if kind is None:
            return None
        name = _node_name(node)
        return (name, kind) if name else None

    # TS / JS family
    kind = _JS_DEFS.get(t)
    if kind is not None:
        name = _node_name(node)
        return (name, kind) if name else None

    if t == "variable_declarator":
        value = node.child_by_field_name("value")
        if value is not None and value.type in _JS_ARROW_VALUES:
            name = _node_name(node)
            return (name, "function") if name else None
    return None


def _chunk_type(raw_kind: str, parent_kind: Optional[str]) -> str:
    """Resolve the final ``Chunk.chunk_type`` from the raw kind + parent."""
    if raw_kind == "class":
        return "class"
    if raw_kind == "method":
        return "method"
    # raw_kind == "function": a function whose immediate definition-parent is a
    # class is a method; otherwise it's a (possibly nested) function.
    return "method" if parent_kind == "class" else "function"


def _span_node(node: Node) -> Node:
    """Widen a definition node to the statement that should bound the chunk.

    * arrow/function-expression: use the enclosing ``lexical_declaration`` so the
      chunk includes ``const foo = ...``.
    * anything wrapped in ``export``: include the ``export`` keyword.
    """
    span = node
    if span.type == "variable_declarator" and span.parent is not None:
        span = span.parent  # lexical_declaration / variable_declaration
    while span.parent is not None and span.parent.type == "export_statement":
        span = span.parent
    return span


def _walk(node: Node, ancestors: tuple[tuple[str, str], ...], ctx: _Context, out: list) -> None:
    """Depth-first traversal collecting definition chunks.

    ``ancestors`` is a tuple of ``(name, kind)`` where ``kind`` is "class" or
    "function" -- only definition scopes, not control-flow blocks -- used to
    build ``qualified_name`` and decide method-ness.
    """
    child_ancestors = ancestors
    detected = _detect_definition(node, ctx)
    if detected is not None:
        name, raw_kind = detected
        parent_kind = ancestors[-1][1] if ancestors else None
        parent_name = ancestors[-1][0] if ancestors else None
        chunk_type = _chunk_type(raw_kind, parent_kind)

        span = _span_node(node)
        qualified = ".".join([a[0] for a in ancestors] + [name])
        out.append(
            Chunk(
                repo_url=ctx.repo_url,
                commit_sha=ctx.commit_sha,
                file_path=ctx.file_path,
                language=ctx.language,
                chunk_type=chunk_type,
                name=name,
                qualified_name=qualified,
                parent_name=parent_name,
                start_line=span.start_point[0] + 1,
                end_line=span.end_point[0] + 1,
                content=ctx.line_text(span.start_point[0], span.end_point[0]),
                docstring=_extract_docstring(node, span, ctx),
            )
        )
        # Descend with this scope pushed.  Methods count as "function" scope for
        # their own children (a function nested in a method is a plain function).
        pushed_kind = "class" if chunk_type == "class" else "function"
        child_ancestors = ancestors + ((name, pushed_kind),)

    for child in node.children:
        _walk(child, child_ancestors, ctx, out)


# --- docstring extraction -------------------------------------------------- #


def _extract_docstring(def_node: Node, span_node: Node, ctx: _Context) -> Optional[str]:
    """Extract a leading docstring/comment for a definition."""
    if ctx.is_python:
        return _python_docstring(def_node, ctx)
    return _js_leading_comment(span_node, ctx)


def _python_docstring(def_node: Node, ctx: _Context) -> Optional[str]:
    """Python: first statement of the body, if it's a string literal."""
    body = def_node.child_by_field_name("body")
    if body is None:
        return None
    first = next((c for c in body.named_children), None)
    if first is None:
        return None
    string_node: Optional[Node] = None
    if first.type == "string":
        string_node = first
    elif first.type == "expression_statement":
        inner = next((c for c in first.named_children), None)
        if inner is not None and inner.type == "string":
            string_node = inner
    if string_node is None:
        return None
    return _clean_python_string(string_node, ctx)


def _clean_python_string(string_node: Node, ctx: _Context) -> str:
    """Return the text inside a Python string node, quotes/prefix stripped."""
    parts = [
        ctx.text(c) for c in string_node.children if c.type == "string_content"
    ]
    if parts:
        return "".join(parts).strip()
    # Fallback: strip surrounding quotes manually.
    raw = ctx.text(string_node)
    return raw.strip().strip("rRbBuUfF").strip("'\"").strip()


def _js_leading_comment(span_node: Node, ctx: _Context) -> Optional[str]:
    """TS/JS: collect the contiguous comment block immediately above a node."""
    comments: list[Node] = []
    prev = span_node.prev_sibling
    expected_line = span_node.start_point[0]  # 0-indexed line just above
    while prev is not None and prev.type == "comment":
        # Only accept comments that butt up against the node (allowing the block
        # to grow upward line by line); stop at the first gap.
        if prev.end_point[0] != expected_line and prev.end_point[0] != expected_line - 1:
            # Allow the immediately-preceding line; anything with a blank-line
            # gap is treated as unrelated.
            if prev.end_point[0] < expected_line - 1:
                break
        comments.append(prev)
        expected_line = prev.start_point[0]
        prev = prev.prev_sibling
    if not comments:
        return None
    comments.reverse()
    return _clean_js_comments([ctx.text(c) for c in comments])


def _clean_js_comments(raw_comments: list[str]) -> str:
    """Strip comment markers from JSDoc / line comments and join."""
    lines: list[str] = []
    for raw in raw_comments:
        text = raw.strip()
        if text.startswith("/*"):
            text = text[2:]
            if text.endswith("*/"):
                text = text[:-2]
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("*"):
                    line = line[1:].strip()
                lines.append(line)
        elif text.startswith("//"):
            lines.append(text[2:].strip())
        else:
            lines.append(text)
    return "\n".join(line for line in lines).strip()


# --- module chunk ---------------------------------------------------------- #


def _is_top_level_def(node: Node, ctx: _Context) -> bool:
    """Whether a top-level child produced its own definition chunk."""
    t = node.type
    if ctx.is_python:
        return t in _PY_DEFS
    if t in ("function_declaration", "generator_function_declaration", "class_declaration"):
        return True
    if t == "export_statement":
        return any(_is_top_level_def(c, ctx) for c in node.named_children)
    if t in ("lexical_declaration", "variable_declaration"):
        return _lexical_is_fn(node)
    return False


def _lexical_is_fn(node: Node) -> bool:
    """A single-declarator ``const/let/var x = () => ...`` (arrow or fn expr)."""
    declarators = [c for c in node.named_children if c.type == "variable_declarator"]
    if len(declarators) != 1:
        return False
    value = declarators[0].child_by_field_name("value")
    return value is not None and value.type in _JS_ARROW_VALUES


def _module_chunk(root: Node, ctx: _Context) -> Optional[Chunk]:
    """Build one ``module`` chunk from top-level code not inside any definition.

    The content is the concatenation of the top-level statements that are *not*
    themselves definitions (imports, module constants, ``if __name__`` guards,
    re-exports, etc.).  These may be non-contiguous, so ``start_line``/
    ``end_line`` span from the first to the last such statement while ``content``
    contains only the kept statements.
    """
    kept: list[Node] = [
        child
        for child in root.named_children
        if child.type != "comment" and not _is_top_level_def(child, ctx)
    ]
    if not kept:
        return None
    # Drop leading/trailing that are pure whitespace-equivalent (comments count).
    content = "\n".join(ctx.text(node) for node in kept).strip()
    if not content:
        return None

    name = _module_name(ctx.file_path)
    docstring = None
    if ctx.is_python and kept:
        docstring = _python_module_docstring(kept[0], ctx)

    return Chunk(
        repo_url=ctx.repo_url,
        commit_sha=ctx.commit_sha,
        file_path=ctx.file_path,
        language=ctx.language,
        chunk_type="module",
        name=name,
        qualified_name=name,
        parent_name=None,
        start_line=kept[0].start_point[0] + 1,
        end_line=kept[-1].end_point[0] + 1,
        content=content,
        docstring=docstring,
    )


def _python_module_docstring(first_stmt: Node, ctx: _Context) -> Optional[str]:
    """Module-level docstring: a string literal as the first top-level stmt."""
    node = first_stmt
    if node.type == "expression_statement":
        inner = next((c for c in node.named_children), None)
        node = inner if inner is not None else node
    if node is not None and node.type == "string":
        return _clean_python_string(node, ctx)
    return None


def _module_name(file_path: str) -> str:
    """A readable module name for a file path (stem, or 'index' fallback)."""
    return PurePosixPath(file_path).stem or PurePosixPath(file_path).name


def _fallback_chunks(
    file_path: str,
    source: str,
    language_hint: str,
    repo_url: str,
    commit_sha: str,
) -> list[Chunk]:
    """A single whole-file ``module`` chunk for unparseable/unsupported files."""
    if not source.strip():
        return []
    language = language_for_path(file_path) or (
        language_hint if language_hint in ("python", "typescript", "javascript") else "python"
    )
    line_count = source.count("\n") + 1
    name = _module_name(file_path)
    return [
        Chunk(
            repo_url=repo_url,
            commit_sha=commit_sha,
            file_path=file_path,
            language=language,  # type: ignore[arg-type]
            chunk_type="module",
            name=name,
            qualified_name=name,
            parent_name=None,
            start_line=1,
            end_line=line_count,
            content=source,
            docstring=None,
        )
    ]
