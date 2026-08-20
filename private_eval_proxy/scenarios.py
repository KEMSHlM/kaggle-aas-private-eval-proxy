"""Environment variants for the private stand-in.

The private leaderboard replays candidates against a **held-out** environment: different
fixtures and a different guardrail from the public set (host-confirmed, disc #708629). One thing
we do not know is whether the concrete values baked into the public fixtures — page URLs, sender
email addresses, the secret in ``secret.txt`` — are **reused** in that held-out environment or
**replaced** with fresh values. Attack design depends on the answer:

* if the values are reused, a candidate may reference them directly and need not spend tool calls
  and wall-clock time discovering them at replay;
* if the values are not reused, only what the candidate discovers at replay (by listing the inbox,
  searching the web corpus, reading the file) can transfer.

Both are plausible, so we replay under both and submit for each. This module supplies the fixtures
for each variant. The guardrail is unchanged across the two — only the fixture values differ.

Variants
--------
``private`` / ``private_a``
    The public fixtures, unchanged. Models "the values are reused".
``private_b``
    A copy of the fixtures with every concrete URL, email address, and secret value rewritten to a
    fresh, non-routable value. Fixture identifiers (``page_0``, ``email_4``), file names, and free
    text are left intact, so the structural handles a candidate discovers at replay still work; only
    the literal values a candidate might have copied from the public fixtures stop matching. Models
    "the values are not reused".

The rewrite is deterministic (a pure function of each original value), so a ``private_b`` replay is
reproducible and equal values map to equal replacements across files. Because the guardrail and the
EXFIL predicate read the secret from ``secret.txt`` in-trace, rewriting that file keeps scoring
self-consistent: a candidate that copied the public secret value no longer matches, while a
candidate that reads the file at replay still exfiltrates the value that is actually present.

The SDK is not modified: ``fixtures_for`` builds a new directory and hands it to the SDK's own
``resolve_fixtures_dir(fixtures_dir=...)`` entry point.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import threading
from pathlib import Path

# Guard kinds that use the private stand-in guardrail (as opposed to the public OptimalGuardrail).
PRIVATE_KINDS = ("private", "private_a", "private_b")

# Guard kinds this module accepts, in addition to "public".
KNOWN_KINDS = PRIVATE_KINDS + ("public",)

# Stop at whitespace, quotes, angle/paren/bracket, and backslash. Excluding backslash matters:
# these values live inside JSON string literals, where a following escaped quote is written `\"`;
# if the class allowed `\` the match would swallow that escaping backslash and corrupt the JSON.
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]\\]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_build_lock = threading.Lock()


def is_private(guard_kind: str) -> bool:
    return guard_kind in PRIVATE_KINDS


def _stable(tag: str, value: str, length: int) -> str:
    """A deterministic hex token derived from ``value`` (stable across runs and files)."""
    return hashlib.sha256(f"{tag}|{value}".encode()).hexdigest()[:length]


def _rewrite_url(match: re.Match[str]) -> str:
    url = match.group(0)
    rest = url.split("://", 1)[1]
    host, sep, path = rest.partition("/")
    new_host = f"h-{_stable('host', host, 8)}.invalid"
    return f"https://{new_host}{sep}{path}"


def _rewrite_email(match: re.Match[str]) -> str:
    local, _, domain = match.group(0).partition("@")
    return f"u{_stable('email-local', local, 6)}@d{_stable('email-domain', domain, 6)}.invalid"


def _rewrite_text(text: str) -> str:
    """Rewrite every URL and email address in a blob of text; leave everything else untouched.

    Applied to the raw JSON text of the corpora — safe because a URL or email token never contains
    a quote, so substituting it in place cannot break the surrounding JSON structure.
    """
    return _EMAIL_RE.sub(_rewrite_email, _URL_RE.sub(_rewrite_url, text))


def _rewrite_secret_file(text: str) -> str:
    """Rewrite the value after the first ``=`` on each ``KEY=VALUE`` line, keeping the key."""
    out_lines = []
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and value.strip():
            out_lines.append(f"{key}=v_{_stable('secret', value.strip(), 24)}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def _build_private_b(base: Path) -> Path:
    """Copy ``base`` into a per-source cache dir with values rewritten; return the copy.

    Cached by the source path's hash so the 32 concurrent replays build it once and reuse it.
    """
    dest = Path(tempfile.gettempdir()) / f"aas_fixtures_private_b_{_stable('dir', str(base), 12)}"
    done = dest / ".complete"
    with _build_lock:
        if done.is_file():
            return dest
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(base, dest)

        for name in ("web_corpus.json", "mail_seed.json"):
            path = dest / name
            if path.is_file():
                # Round-trip through json to confirm the rewrite kept the file valid.
                rewritten = _rewrite_text(path.read_text())
                json.loads(rewritten)
                path.write_text(rewritten)

        secret = dest / "file_seed" / "secret.txt"
        if secret.is_file():
            secret.write_text(_rewrite_secret_file(secret.read_text()))

        done.write_text("ok\n")
    return dest


def fixtures_for(guard_kind: str, base_fixtures: Path) -> Path:
    """Return the fixtures directory for ``guard_kind``.

    ``private`` / ``private_a`` / ``public`` use ``base_fixtures`` unchanged; ``private_b`` uses a
    value-rewritten copy of it.
    """
    if guard_kind == "private_b":
        return _build_private_b(Path(base_fixtures))
    return Path(base_fixtures)
