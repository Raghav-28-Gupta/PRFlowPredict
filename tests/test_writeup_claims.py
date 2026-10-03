"""Every result the write-up cites, checked against its committed artifact and its document.

See writeup_claims.py for the registry, and the Phase 8 spec (section 7) for the rules."""
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


def test_every_claim_source_is_committed():
    """A reader who clones the repo must be able to check every number (spec section 7.3)."""
    out = subprocess.run(["git", "ls-files"], cwd=wc.ROOT, capture_output=True, text=True, check=True)
    tracked = set(out.stdout.splitlines())
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
