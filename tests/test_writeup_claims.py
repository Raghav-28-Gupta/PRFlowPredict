"""Every result the write-up cites, checked against its committed artifact and its document.

See writeup_claims.py for the registry, and the Phase 8 spec (section 7) for the rules."""
import ast
import re
import subprocess

import pytest

import writeup_claims as wc

DOCS = (wc.README, wc.REPORT)
LINKED_DOCS = DOCS + ("docs/data_collection.md",)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _doc(rel: str) -> str:
    """Whitespace-normalised, so a claim that Markdown wraps across two lines still counts.
    Reads wc.ROOT at call time so tests can point it at a temporary directory."""
    return re.sub(r"\s+", " ", (wc.ROOT / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the claims, both directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim", wc.CLAIMS, ids=lambda c: c.name)
def test_claim_recomputes_to_its_registered_string(claim):
    """Artifact direction: if a committed artifact changes, the registered string goes stale."""
    assert claim.render() == claim.expected


@pytest.mark.parametrize("claim", wc.CLAIMS, ids=lambda c: c.name)
def test_claim_appears_in_every_listed_document(claim):
    """Document direction: a mistyped, missing or stale number in the prose fails here."""
    for doc in claim.docs:
        assert claim.expected in _doc(doc), f"{claim.expected!r} is missing from {doc}"


def test_claim_names_are_unique():
    names = [c.name for c in wc.CLAIMS]
    assert len(names) == len(set(names))


def _tracked_files(root) -> set[str] | None:
    """git's tracked files under root, or None outside a git checkout (e.g. a GitHub
    'Download ZIP' copy) or without git, where the committed-source check cannot run."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return set(out.stdout.splitlines()) if out.returncode == 0 else None


def test_tracked_files_is_none_outside_a_git_checkout(tmp_path):
    assert _tracked_files(tmp_path) is None


def test_every_claim_source_is_committed():
    """A reader who clones the repo must be able to check every number (spec section 7.3)."""
    tracked = _tracked_files(wc.ROOT)
    if tracked is None:
        pytest.skip("not a git checkout (e.g. a ZIP download), so committed files cannot be listed")
    untracked = sorted({s for c in wc.CLAIMS for s in c.sources} - tracked)
    assert not untracked, f"claims read files a clone would not have: {untracked}"


def test_parse_demands_exactly_one_match(tmp_path, monkeypatch):
    """If a regenerated phase document rewords a sentence, the claim must fail loudly rather
    than silently match a different sentence or nothing."""
    (tmp_path / "d.md").write_text("value is 1.5\nvalue is 2.5\n", encoding="utf-8")
    monkeypatch.setattr(wc, "ROOT", tmp_path)
    with pytest.raises(LookupError):
        wc._parse("d.md", r"value is ([\d.]+)")          # two matches
    with pytest.raises(LookupError):
        wc._parse("d.md", r"total is ([\d.]+)")          # no match
    assert wc._parse("d.md", r"value is (2\.[\d]+)") == (2.5,)


def test_wrapped_claim_still_counts(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_text("a result of 0.769\n[0.669, 0.856] here\n", encoding="utf-8")
    monkeypatch.setattr(wc, "ROOT", tmp_path)
    assert "0.769 [0.669, 0.856]" in _doc("x.md")


# ---------------------------------------------------------------------------
# honesty constraints that carry no number (spec section 9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc,phrase", wc.REQUIRED_PHRASES, ids=lambda v: str(v)[:30])
def test_required_honesty_phrase_present(doc, phrase):
    assert phrase in _doc(doc)


@pytest.mark.parametrize("pattern", wc.FORBIDDEN_PATTERNS)
@pytest.mark.parametrize("doc", DOCS)
def test_forbidden_phrasing_absent(doc, pattern):
    hit = re.search(pattern, _doc(doc), flags=re.IGNORECASE)
    assert hit is None, f"{doc} says {hit.group(0)!r}"


@pytest.mark.parametrize("pattern,reason", wc.RETRACTED, ids=lambda v: str(v)[:30])
@pytest.mark.parametrize("doc", DOCS)
def test_retracted_phrasing_absent(doc, pattern, reason):
    hit = re.search(pattern, _doc(doc), flags=re.IGNORECASE)
    assert hit is None, f"{doc} says {hit.group(0)!r}, but {reason}"


@pytest.mark.parametrize("pattern", wc.BLUEPRINT_6B_UPGRADES)
def test_blueprint_never_upgrades_6b(pattern):
    hit = re.search(pattern, _doc(wc.BLUEPRINT), flags=re.IGNORECASE)
    assert hit is None, f"{wc.BLUEPRINT} says {hit.group(0)!r}"


def test_writeup_modules_parse_on_python_311():
    """Python 3.11 and earlier reject a backslash inside an f-string's braces (PEP 701 lifted
    that in 3.12), and ast.parse(feature_version=...) does not detect it, so look directly."""
    for mod in ("writeup_claims.py", "writeup_figures.py"):
        src = (wc.ROOT / mod).read_text(encoding="utf-8")
        bad = [seg for node in ast.walk(ast.parse(src)) if isinstance(node, ast.JoinedStr)
               for part in node.values if isinstance(part, ast.FormattedValue)
               for seg in [ast.get_source_segment(src, part.value) or ""] if "\\" in seg]
        assert not bad, f"{mod}: backslash inside f-string braces: {bad}"


# ---------------------------------------------------------------------------
# a reader clicking around must not hit a dead link
# ---------------------------------------------------------------------------

def _slug(heading: str) -> str:
    """GitHub's heading anchor: lower-case, punctuation dropped, spaces to hyphens."""
    s = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s", "-", s)


@pytest.mark.parametrize("doc", LINKED_DOCS)
def test_relative_links_resolve(doc):
    base = (wc.ROOT / doc).parent
    broken = []
    for target in LINK.findall((wc.ROOT / doc).read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        full = (base / path).resolve() if path else (wc.ROOT / doc).resolve()
        if not full.exists():
            broken.append(target)
            continue
        if anchor:
            heads = [ln.lstrip("#").strip() for ln in full.read_text(encoding="utf-8").splitlines()
                     if ln.startswith("#")]
            if anchor not in {_slug(h) for h in heads}:
                broken.append(target)
    assert not broken, f"{doc}: broken links {broken}"
