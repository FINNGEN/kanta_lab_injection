#!/usr/bin/env python3
"""
build.py — build the whole site.

  <out>/                 primary injection results (index.html, doc.html, tests/)   from data/
  <out>/unharmonized/    unharmonized suggestions portal                             from data/unharmonized/

The unharmonized section is built only when data/unharmonized/groups.tsv exists; the
primary pages then get an "Unharmonized" link in their nav.

Review issues: the portal's group pages open GitHub issues titled "[unharm] <test> · <unit>" in
FINNGEN/kanta_lab_harmonisation_public (--issue-repo); their figures are served from this repo. Their status is read here from the GitHub API (token from
GITHUB_TOKEN when set, e.g. in Actions), or from --issues-json for local tests; when
neither works the pages just show "no issue yet" and check again live in the browser.

  python3 scripts/site/build.py                                   # data/ → html/
  python3 scripts/site/build.py --out-dir _site --site-url https://finngen.github.io/kanta_lab_injection/dev/
  python3 scripts/site/build.py --unharmonized-dir /tmp/x/out --out-dir /tmp/site --no-github
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import primary        # noqa: E402
import unharmonized   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = "FINNGEN/kanta_lab_injection"                    # this repo: serves the issue figures
DEFAULT_ISSUE_REPO = "FINNGEN/kanta_lab_harmonisation_public"    # where review issues are opened
DEFAULT_SITE = "https://finngen.github.io/kanta_lab_injection/"


def fetch_issues(repo, prefix="[unharm] ", pages=10, timeout=15):
    """title -> {number, state, url, sid} for every issue in `repo` whose title starts with the prefix
    (open and closed); sid is the suggestion id written in the issue body, to tell when a suggestion
    has changed since. Searched by title, so the repo needs no special label."""
    out = {}
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "kanta-lab-injection-site"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    q = urllib.parse.quote(f'repo:{repo} is:issue in:title "{prefix.strip()}"')
    for page in range(1, pages + 1):
        url = f"https://api.github.com/search/issues?q={q}&per_page=100&page={page}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            items = json.loads(r.read()).get("items", [])
        for it in items:
            if "pull_request" in it or not it["title"].startswith(prefix):
                continue
            m = re.search(r"Suggestion id: `([0-9a-f]+)`", it.get("body") or "")
            out.setdefault(it["title"], {"number": it["number"], "state": it["state"], "url": it["html_url"],
                                         "sid": m.group(1) if m else ""})
        if len(items) < 100:
            break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Primary release files")
    p.add_argument("--unharmonized-dir", type=Path, default=None,
                   help="Unharmonized suggestions (default: <data-dir>/unharmonized)")
    p.add_argument("--readme", type=Path, default=ROOT / "README.md")
    p.add_argument("--out-dir", type=Path, default=ROOT / "html")
    p.add_argument("--repo", default=DEFAULT_REPO, help="This repo on GitHub (serves the issue figures)")
    p.add_argument("--issue-repo", default=DEFAULT_ISSUE_REPO, help="GitHub repo where review issues are opened")
    p.add_argument("--site-url", default=DEFAULT_SITE, help="Public URL of this site (links in issues)")
    p.add_argument("--branch", default="dev",
                   help="Branch whose data/unharmonized/figures/ the issue images are served from")
    p.add_argument("--issues-json", type=Path, default=None,
                   help="Issue statuses from a file instead of GitHub: {title: {number, state, url}}")
    p.add_argument("--no-github", action="store_true", help="Do not ask GitHub for issue statuses")
    p.add_argument("--mock-note", default="", help="Banner shown on every portal page (e.g. for synthetic data)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    unharm_dir = args.unharmonized_dir or args.data_dir / "unharmonized"
    has_unharm = (unharm_dir / "groups.tsv").exists()

    primary.NAV_UNHARM = has_unharm
    primary.build_index(args.out_dir, args.data_dir)
    primary.build_doc(args.out_dir, args.readme)
    primary.build_test_pages(args.out_dir, args.data_dir)

    if has_unharm:
        issues, at = {}, "not checked"
        if args.issues_json:
            issues = json.loads(args.issues_json.read_text())
            at = f"{args.issues_json.name}"
        elif not args.no_github:
            try:
                issues = fetch_issues(args.issue_repo)
                at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            except Exception as e:      # offline, rate-limited, private repo without token
                print(f"Issue statuses not available ({e}); pages will check live.")
        fig_base = f"https://raw.githubusercontent.com/{args.repo}/{args.branch}/data/unharmonized/figures/"
        unharmonized.build(args.out_dir / "unharmonized", unharm_dir, args.issue_repo, args.site_url, issues, at,
                           mock_note=args.mock_note, fig_base=fig_base)


if __name__ == "__main__":
    main()
