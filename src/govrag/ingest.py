"""Fetch GOV.UK employment guidance from the public Content API.

Each guide "part" becomes one document. Part URLs are stable, so they double
as citation targets and as gold labels in the eval set.

GOV.UK content is published under the Open Government Licence v3.0.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

API = "https://www.gov.uk/api/content/"

GUIDES = [
    "holiday-entitlement-rights",
    "statutory-sick-pay",
    "maternity-pay-leave",
    "paternity-pay-leave",
    "shared-parental-leave-and-pay",
    "adoption-pay-leave",
    "parental-leave",
    "time-off-for-dependants",
    "your-right-to-minimum-wage",
    "national-minimum-wage-rates",
    "employment-contracts-and-conditions",
    "dismissal",
    "redundant-your-rights",
    "flexible-working",
    "maximum-weekly-working-hours",
    "rest-breaks-work",
    "understanding-your-pay",
    "employment-status",
    "workplace-pensions",
    "solve-workplace-dispute",
    "whistleblowing",
    "discrimination-your-rights",
    "agency-workers-your-rights",
    "contract-types-and-employer-responsibilities",
    "handing-in-your-notice",
    "employment-tribunals",
    "jury-service",
    "join-trade-union",
    "taking-disciplinary-action",
]


def html_to_text(html: str) -> str:
    """Flatten GOV.UK HTML to text, keeping headings as '## ' lines and lists as '- '."""
    soup = BeautifulSoup(html, "html.parser")
    lines: list[str] = []

    def walk(node: Tag) -> None:
        for el in node.children:
            if isinstance(el, NavigableString):
                text = el.strip()
                if text:
                    lines.append(text)
                continue
            if not isinstance(el, Tag):
                continue
            name = el.name
            if name in ("h2", "h3", "h4"):
                lines.append(f"## {el.get_text(' ', strip=True)}")
            elif name in ("p", "summary"):
                lines.append(el.get_text(" ", strip=True))
            elif name == "li":
                lines.append(f"- {el.get_text(' ', strip=True)}")
            elif name == "tr":
                cells = [c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])]
                lines.append(" | ".join(cells))
            else:
                walk(el)

    walk(soup)
    return "\n".join(line for line in lines if line)


def fetch_guide(client: httpx.Client, slug: str) -> list[dict]:
    resp = client.get(API + slug)
    resp.raise_for_status()
    item = resp.json()
    if item.get("schema_name") == "redirect":
        # Renamed guides keep a redirect at the old slug; follow it once.
        slug = item["redirects"][0]["destination"].strip("/")
        return fetch_guide(client, slug)
    details = item["details"]
    updated = item.get("public_updated_at", "")[:10]

    if "parts" in details:
        parts = [(p["slug"], p["title"], p["body"]) for p in details["parts"]]
    else:
        parts = [("", item["title"], details.get("body", ""))]

    docs = []
    for part_slug, part_title, body in parts:
        doc_id = f"{slug}/{part_slug}" if part_slug else slug
        docs.append(
            {
                "doc_id": doc_id,
                "guide": slug,
                "title": item["title"],
                "part_title": part_title.strip(),
                "url": f"https://www.gov.uk/{doc_id}",
                "updated": updated,
                "text": html_to_text(body),
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    args = parser.parse_args()

    docs: list[dict] = []
    with httpx.Client(timeout=20, headers={"User-Agent": "govuk-rag (portfolio project)"}) as client:
        for slug in GUIDES:
            try:
                fetched = fetch_guide(client, slug)
            except httpx.HTTPStatusError as err:
                print(f"skip {slug}: HTTP {err.response.status_code}")
                continue
            docs.extend(fetched)
            print(f"{slug}: {len(fetched)} parts")
            time.sleep(0.2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} documents to {args.out}")


if __name__ == "__main__":
    main()
