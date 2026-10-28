#!/usr/bin/env python3
"""The tombstone record is permanent (Andrew, 2026-10-28).

docs/TOMBSTONES.md entries, their patches under experiments/removed/ and the inline tombstone
markers in the code are never deleted, rewritten or squashed; deletions of experiments are
only ever added. This test is the fence for that rule (tools/hooks/pre-push runs it on every
pushed branch). Plain unittest, reads text files only, imports nothing from the product.

For every row of the TOMBSTONES.md index it checks:
  (a) the section the row links to exists;
  (b) every patch the row links to exists under experiments/removed/ (rows whose code never
      reached main say "lane only" or "no code" and link no patch; their section must name
      the branch@sha or say why there is no code, or be listed in experiments/tombstones/GAPS.md);
  (c) at least one inline tombstone marker (a '#' comment starting with TOMBSTONE: and naming
      the define, in a .mojo, .py, .pyi or .sh file) exists in the tree, unless the row's code
      never reached main or the row is listed in experiments/tombstones/pre_rule_entries.tsv
      (entries that predate the 2026-10-10 inline rule, or recorded gaps; that file only shrinks).
It also checks that the index never has fewer rows than experiments/tombstones/index_count.txt
(the committed number only goes up), and, against the merge-base with main, that every index
row, section heading, patch file and inline marker define of the base is still present
unchanged, that index_count.txt did not go down and that the allowlist did not grow.

Usage:
  python3 tools/tests/test_tombstones_preserved.py                 # the working tree
  python3 tools/tests/test_tombstones_preserved.py --tree <sha>    # a commit (pre-push)
  ... [--base <sha> | --base none]   default base: merge-base with origin/main (or main)
"""
import argparse
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

DOC = 'docs/TOMBSTONES.md'
REMOVED = 'experiments/removed'
COUNT_FILE = 'experiments/tombstones/index_count.txt'
ALLOW_FILE = 'experiments/tombstones/pre_rule_entries.tsv'
GAPS_FILE = 'experiments/tombstones/GAPS.md'
SELF = 'tools/tests/test_tombstones_preserved.py'
MARKER_GLOBS = ['*.mojo', '*.py', '*.pyi', '*.sh']
# a '#' comment whose text starts with the marker word (kept split so this file never matches itself)
MARKER_WORD = 'TOMB' + 'STONE:'
MARKER_RE = re.compile(r'#\s*' + MARKER_WORD + r'\s*(.*)')
GREP_PATTERN = '#[[:space:]]*' + MARKER_WORD
GREP_ARGS = ['-I', '-n', '-z', '-A', '4']   # the marker line and the comment lines continuing it

OPTS = argparse.Namespace(tree=None, base=None, repo=None)


def _git(repo, *args, check=True, binary=False):
    out = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, check=False)
    if check and out.returncode != 0:
        raise RuntimeError('git %s failed: %s' % (' '.join(args), out.stderr.decode(errors='replace')[:400]))
    if out.returncode != 0:
        return None
    return out.stdout if binary else out.stdout.decode('utf-8', errors='replace')


class WorkTree:
    """The checkout on disk (the working copies of tracked files: git add a new file before running)."""

    def __init__(self, root):
        self.root = Path(root)
        self.label = 'working tree %s' % self.root

    def read(self, path):
        p = self.root / path
        return p.read_text(encoding='utf-8', errors='replace') if p.is_file() else None

    def exists(self, path):
        return (self.root / path).is_file()

    def files_under(self, prefix):
        d = self.root / prefix
        if not d.is_dir():
            return []
        return sorted(str(p.relative_to(self.root)) for p in d.rglob('*') if p.is_file())

    def blob_ids(self, paths):
        paths = [p for p in paths if self.exists(p)]
        if not paths:
            return {}
        out = subprocess.run(['git', '-C', str(self.root), 'hash-object', '--stdin-paths'],
                             input='\n'.join(paths) + '\n', capture_output=True, text=True, check=True)
        return dict(zip(paths, out.stdout.split()))

    def marker_lines(self):
        out = _git(self.root, 'grep', *GREP_ARGS, '-e', GREP_PATTERN, '--', *MARKER_GLOBS, check=False)
        return _split_grep(out or '', prefix='')


class CommitTree:
    """A commit, read through git (what the pre-push hook judges)."""

    def __init__(self, repo, sha):
        self.repo = repo
        self.sha = sha
        self.label = 'commit %s' % sha

    def read(self, path):
        data = _git(self.repo, 'cat-file', '-p', '%s:%s' % (self.sha, path), check=False, binary=True)
        return None if data is None else data.decode('utf-8', errors='replace')

    def exists(self, path):
        return _git(self.repo, 'cat-file', '-e', '%s:%s' % (self.sha, path), check=False) is not None

    def _ls(self, prefix):
        out = _git(self.repo, 'ls-tree', '-r', self.sha, '--', prefix + '/', check=False) or ''
        rows = {}
        for line in out.splitlines():
            meta, _, path = line.partition('\t')
            if meta.split()[1] == 'blob':
                rows[path] = meta.split()[2]
        return rows

    def files_under(self, prefix):
        return sorted(self._ls(prefix))

    def blob_ids(self, paths):
        ids = {}
        for prefix in sorted({p.rsplit('/', 1)[0] for p in paths}):
            ids.update(self._ls(prefix))
        return {p: ids[p] for p in paths if p in ids}

    def marker_lines(self):
        out = _git(self.repo, 'grep', *GREP_ARGS, '-e', GREP_PATTERN, self.sha, '--', *MARKER_GLOBS, check=False)
        return _split_grep(out or '', prefix=self.sha + ':')


def _split_grep(out, prefix):
    """Marker blocks: the marker line plus the comment lines that continue it (git grep -n -z -A)."""
    blocks = []
    cur = None          # [path, last line number, texts]
    for line in out.split('\n'):
        if line == '--' or not line:
            cur = None
            continue
        parts = line.split('\0', 2)
        if len(parts) != 3:
            cur = None
            continue
        path, num, text = parts
        if prefix and path.startswith(prefix):
            path = path[len(prefix):]
        if path == SELF:
            continue
        num = int(num)
        m = MARKER_RE.search(text)
        if m:
            cur = [path, num, [m.group(1).strip()]]
            blocks.append(cur)
            continue
        stripped = text.strip()
        if cur and cur[0] == path and num == cur[1] + 1 and stripped.startswith('#'):
            cur[1] = num
            cur[2].append(stripped.lstrip('#').strip())
        else:
            cur = None
    return [(b[0], ' '.join(b[2])) for b in blocks]


# ---- TOMBSTONES.md parsing ----

def _slug(text):
    text = re.sub(r'<[^>]+>', '', text).strip().lower()
    text = re.sub(r'[^\w\- ]', '', text)
    return text.replace(' ', '-')


def headings(doc):
    """Every heading line (outside code fences) and the GitHub anchors they get."""
    seen = {}
    anchors = {}
    lines = doc.split('\n')
    fence = False
    for i, line in enumerate(lines):
        if line.startswith('```'):
            fence = not fence
            continue
        if fence:
            continue
        m = re.match(r'^(#{1,6})\s+(.*?)\s*#*\s*$', line)
        if m:
            base = _slug(m.group(2))
            n = seen.get(base, 0)
            seen[base] = n + 1
            anchors[base if n == 0 else '%s-%d' % (base, n)] = (i, len(m.group(1)))
        for a in re.findall(r'<a\s+(?:id|name)="([^"]+)"', line):
            anchors.setdefault(a, (i, 7))
    return anchors, lines


def section_text(lines, start, level):
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r'^(#{1,6})\s', lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    return '\n'.join(lines[start:end])


def heading_lines(doc):
    return [line for line in doc.split('\n') if re.match(r'^#{1,6}\s', line)]


def index_rows(doc):
    """The rows of the '## Index' table, as raw lines."""
    rows = []
    inside = False
    for line in doc.split('\n'):
        if line.strip() == '## Index':
            inside = True
            continue
        if inside and line.startswith('## '):
            break
        if inside and line.startswith('| ') and not line.startswith('| define |'):
            rows.append(line)
    return rows


def split_row(line):
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    return cells


def row_kind(patch_cell):
    if re.search(r'\]\(\.\./' + REMOVED + r'/[^)]+\)', patch_cell):
        return 'patch'
    low = patch_cell.lower()
    if low.startswith('lane only'):
        return 'lane'
    if low.startswith('no code'):
        return 'nocode'
    if low.startswith('none'):
        return 'none'
    return 'unknown'


def row_key(name_cell):
    m = re.search(r'`([^`]+)`', name_cell)
    if m:
        return m.group(1)
    m = re.search(r'\[([^\]]+)\]', name_cell)
    return (m.group(1) if m else name_cell).strip()


def row_anchors(name_cell):
    return re.findall(r'\]\(#([^)]+)\)', name_cell)


def row_patches(patch_cell):
    return re.findall(r'\]\(\.\./(' + REMOVED + r'/[^)]+)\)', patch_cell)


def row_tokens(name_cell, patch_cell):
    tokens = set(re.findall(r'`([^`]+)`', name_cell))
    tokens.update(re.findall(r'MOJOLEARN_[A-Z0-9_]+', name_cell))
    for p in row_patches(patch_cell):
        stem = p.rsplit('/', 1)[1]
        stem = re.sub(r'\.patch$', '', stem)
        tokens.add(stem)
        tokens.add(re.sub(r'-arm\d+$', '', stem))
    m = re.search(r'\[([^\]]+)\]', name_cell)
    plain = m.group(1) if m else name_cell
    if '`' not in plain:
        tokens.add(plain.strip())
    # snake_case names in a prose row ("lamb_step and the rest of the x_sequence batch")
    tokens.update(w.strip('./') for w in re.findall(r'[A-Za-z0-9_*./]*_[A-Za-z0-9_*./]*', plain))
    return {t.strip() for t in tokens if len(t.strip()) >= 4}


_WORDS = {}


def token_marked(token, marker_text):
    """The token appears in the marker text, not as part of a longer identifier."""
    if re.fullmatch(r'[A-Za-z0-9_]+', token):
        words = _WORDS.get(id(marker_text))
        if words is None or words[0] is not marker_text:
            words = (marker_text, set(re.findall(r'[A-Za-z0-9_]+', marker_text)))
            _WORDS[id(marker_text)] = words
        return token in words[1]
    pat = r'(?<![A-Za-z0-9_])' + re.escape(token) + r'(?![A-Za-z0-9_])'
    return re.search(pat, marker_text) is not None


def read_allowlist(text):
    rows = []
    for line in (text or '').split('\n'):
        if not line.strip() or line.startswith('#'):
            continue
        cells = line.split('\t')
        rows.append(tuple(c.strip() for c in cells[:2]))
    return rows


def marker_defines(marker_lines):
    names = set()
    for _, rest in marker_lines:
        m = re.match(r'`?([A-Za-z_][A-Za-z0-9_.*]*)', rest)
        if m:
            names.add(m.group(1))
    return names


# ---- the tests ----

_TREES = {}


def _cached(tree):
    """One marker grep per tree per run."""
    lines = tree.marker_lines()
    tree.marker_lines = lambda: lines
    return tree


def _tree():
    if 'head' not in _TREES:
        _TREES['head'] = _cached(CommitTree(OPTS.repo, OPTS.tree) if OPTS.tree else WorkTree(OPTS.repo))
    return _TREES['head']


def _base():
    if OPTS.base == 'none':
        return None
    if OPTS.base:
        return _cached(CommitTree(OPTS.repo, OPTS.base))
    head = OPTS.tree or 'HEAD'
    for main in ('origin/main', 'main'):
        sha = _git(OPTS.repo, 'merge-base', head, main, check=False)
        if sha and sha.strip():
            return _cached(CommitTree(OPTS.repo, sha.strip()))
    return None


class TombstonesPreserved(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.tree = _tree()
        cls.doc = cls.tree.read(DOC)
        if cls.doc is None:
            raise AssertionError('%s is missing from the %s' % (DOC, cls.tree.label))
        cls.rows = index_rows(cls.doc)
        cls.anchors, cls.lines = headings(cls.doc)
        cls.markers = cls.tree.marker_lines()
        cls.marker_text = '\n'.join(rest for _, rest in cls.markers)
        cls.allow = read_allowlist(cls.tree.read(ALLOW_FILE))
        cls.gaps = set(re.findall(r'^\| `([^`]+)` \|', cls.tree.read(GAPS_FILE) or '', re.M))

    def test_index_has_rows(self):
        self.assertGreater(len(self.rows), 0, 'no index rows parsed from %s' % DOC)

    def test_rows_are_well_formed(self):
        bad = [r[:160] for r in self.rows if len(split_row(r)) != 5]
        self.assertEqual(bad, [], 'index rows without 5 cells')
        bad = [r[:160] for r in self.rows if row_kind(split_row(r)[4]) == 'unknown']
        self.assertEqual(bad, [], 'index rows whose patch cell is neither a patch link nor '
                                  '"lane only" / "no code" / "none (recoverable at <sha>)"')

    def test_a_sections_exist(self):
        missing = []
        for r in self.rows:
            name = split_row(r)[0]
            anchors = row_anchors(name)
            if not anchors:
                # an unlinked row names its section by its heading text ("<heading> (no define)")
                if _slug(re.sub(r'\s*\([^)]*\)\s*$', '', name)) not in self.anchors:
                    missing.append('no section link and no heading named like the row: ' + r[:160])
            for a in anchors:
                if a not in self.anchors:
                    missing.append('#%s (row %s)' % (a, row_key(name)))
        self.assertEqual(missing, [], 'index rows whose section is missing')

    def test_b_patches_exist(self):
        missing = []
        for r in self.rows:
            cells = split_row(r)
            for p in row_patches(cells[4]):
                if not self.tree.exists(p):
                    missing.append('%s (row %s)' % (p, row_key(cells[0])))
        self.assertEqual(missing, [], 'index rows whose patch file is missing under %s' % REMOVED)

    def test_b_unpatched_rows_say_where_the_code_is(self):
        """A row with no patch must name the recoverable branch@sha / sha, or say there was no code."""
        bad = []
        for r in self.rows:
            cells = split_row(r)
            kind = row_kind(cells[4])
            if kind not in ('lane', 'nocode', 'none'):
                continue
            if kind == 'none' and re.search(r'`[0-9a-f]{7,40}`', cells[4]):
                continue
            texts = []
            for a in row_anchors(cells[0]):
                if a in self.anchors:
                    i, level = self.anchors[a]
                    texts.append(section_text(self.lines, i, level))
            body = '\n'.join(texts)
            if kind == 'nocode' and re.search(r'no code|never (?:reached|landed|committed)|not committed', body, re.I):
                continue
            if re.search(r'(?<![0-9a-z])[0-9a-f]{7,40}(?![0-9a-z])', body):
                continue
            if row_key(cells[0]) in self.gaps:
                continue      # unrecoverable, recorded in GAPS_FILE
            bad.append(row_key(cells[0]))
        self.assertEqual(bad, [], 'unpatched rows whose section names no recoverable sha')

    def test_c_inline_markers(self):
        allowed = set(self.allow)
        missing = []
        for r in self.rows:
            cells = split_row(r)
            kind = row_kind(cells[4])
            if kind in ('lane', 'nocode'):
                continue          # the code never reached main: there is no deletion site to mark
            tokens = row_tokens(cells[0], cells[4])
            if any(token_marked(t, self.marker_text) for t in tokens):
                continue
            key = row_key(cells[0])
            anchor = (row_anchors(cells[0]) or [''])[0]
            if (key, anchor) in allowed:
                continue
            missing.append('%s\t%s' % (key, anchor))
        self.assertEqual(missing, [], 'index rows with no inline %s marker in %s files and no row in %s'
                         % (MARKER_WORD, '/'.join(MARKER_GLOBS), ALLOW_FILE))

    def test_c_allowlist_rows_are_live(self):
        """Every allowlist row names an index row that still has no marker (so the list only shrinks)."""
        live = {}
        for r in self.rows:
            cells = split_row(r)
            key = row_key(cells[0])
            anchor = (row_anchors(cells[0]) or [''])[0]
            marked = any(token_marked(t, self.marker_text) for t in row_tokens(cells[0], cells[4]))
            live[(key, anchor)] = marked
        stale = ['%s\t%s' % k for k in self.allow if k not in live]
        self.assertEqual(stale, [], 'allowlist rows that match no index row')
        marked = ['%s\t%s' % k for k in self.allow if live.get(k)]
        self.assertEqual(marked, [], 'allowlist rows whose entry now has an inline marker: delete the row')
        self.assertEqual(len(self.allow), len(set(self.allow)), 'duplicate allowlist rows')

    def test_index_count_never_decreases(self):
        text = self.tree.read(COUNT_FILE)
        self.assertIsNotNone(text, '%s is missing' % COUNT_FILE)
        nums = [line for line in text.split('\n') if line.strip() and not line.startswith('#')]
        self.assertEqual(len(nums), 1, '%s must hold one number' % COUNT_FILE)
        pinned = int(nums[0].strip())
        self.assertLessEqual(pinned, len(self.rows),
                             'TOMBSTONES.md index has %d rows but %s pins %d: an index row was removed'
                             % (len(self.rows), COUNT_FILE, pinned))


class TombstonesKeptSinceBase(unittest.TestCase):
    maxDiff = None
    """Nothing the merge-base with main recorded is gone or rewritten."""

    @classmethod
    def setUpClass(cls):
        cls.base = _base()
        if cls.base is None:
            raise unittest.SkipTest('no base commit (no main ref, or --base none)')
        cls.tree = _tree()
        cls.base_doc = cls.base.read(DOC) or ''
        cls.doc = cls.tree.read(DOC) or ''

    def test_base_index_rows_kept_verbatim(self):
        head = {}
        for r in index_rows(self.doc):
            head[r] = head.get(r, 0) + 1
        gone = []
        for r in index_rows(self.base_doc):
            if head.get(r, 0) > 0:
                head[r] -= 1
            else:
                gone.append(r[:200])
        self.assertEqual(gone, [], 'index rows of base %s removed or rewritten' % self.base.sha)

    def test_base_headings_kept(self):
        head = set(heading_lines(self.doc))
        gone = [h for h in heading_lines(self.base_doc) if h not in head]
        self.assertEqual(gone, [], 'section headings of base %s removed or renamed' % self.base.sha)

    def test_base_patches_kept_byte_identical(self):
        base_files = self.base.files_under(REMOVED)
        base_ids = self.base.blob_ids(base_files)
        head_ids = self.tree.blob_ids(base_files)
        gone = [p for p in base_files if p not in head_ids]
        changed = [p for p in base_files if p in head_ids and head_ids[p] != base_ids.get(p)]
        self.assertEqual(gone, [], 'patches of base %s deleted' % self.base.sha)
        self.assertEqual(changed, [], 'patches of base %s rewritten' % self.base.sha)

    def test_base_marker_defines_kept(self):
        head = marker_defines(self.tree.marker_lines())
        gone = sorted(marker_defines(self.base.marker_lines()) - head)
        self.assertEqual(gone, [], 'inline tombstone markers of base %s deleted' % self.base.sha)

    def test_count_and_allowlist_monotone(self):
        def num(t):
            if t is None:
                return None
            vals = [line.strip() for line in t.split('\n') if line.strip() and not line.startswith('#')]
            return int(vals[0]) if vals else None
        b, h = num(self.base.read(COUNT_FILE)), num(self.tree.read(COUNT_FILE))
        if b is not None:
            self.assertIsNotNone(h, '%s deleted' % COUNT_FILE)
            self.assertGreaterEqual(h, b, '%s went down from %d to %d' % (COUNT_FILE, b, h))
        base_allow = self.base.read(ALLOW_FILE)
        if base_allow is not None:
            grown = sorted(set(read_allowlist(self.tree.read(ALLOW_FILE))) - set(read_allowlist(base_allow)))
            self.assertEqual(grown, [], '%s grew (it only shrinks)' % ALLOW_FILE)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--tree')
    ap.add_argument('--base')
    ap.add_argument('--repo')
    opts, rest = ap.parse_known_args()
    if opts.repo:
        repo = opts.repo
    elif opts.tree:
        repo = os.getcwd()
    else:
        repo = str(Path(__file__).resolve().parents[2])
    top = _git(repo, 'rev-parse', '--show-toplevel', check=False)
    OPTS.repo = top.strip() if top else repo
    OPTS.tree, OPTS.base = opts.tree, opts.base
    unittest.main(argv=[sys.argv[0]] + rest, verbosity=1)


if __name__ == '__main__':
    main()
