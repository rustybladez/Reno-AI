"""One-time helper: PDF -> DRAFT policy.json.

The output is a draft. A human must check it against the PDF before it is used
(see PRD §6). The app never parses the PDF at runtime.

Usage:
    uv run python scripts/extract_policy.py "XYZ Company Leave Policy (Sample).pdf"
    uv run python scripts/extract_policy.py <pdf> --out data/policy.draft.json

What it does:
1. Reads each page's text blocks and tables, in top-to-bottom order, so tables land
   under the heading they visually belong to (raw text extraction puts them elsewhere).
2. Drops page headers/footers.
3. Splits the text into one entry per clause: "2.2", "4.1(d)", or a whole section ("1").
4. Prints a word-coverage report: every word in the PDF should appear in the draft.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

FOOTER_PATTERNS = [
    re.compile(r"^XYZ Company Leave Policy \(Sample\)$"),
    re.compile(r"^Page \d+ of \d+$"),
]
END_MARKER = re.compile(r"^End of policy\.")

TOP_HEADING = re.compile(r"^(\d+)\.\s*([A-Z].*)$")  # "5. Weekly Holidays and Public Holidays"
# "4.1 Annual Leave" or "2.1 This policy applies...". Needs a space, so a wrapped "4.1)." isn't a clause.
NUMBERED = re.compile(r"^(\d+\.\d+)\s+(\S.*)$")
LETTERED = re.compile(r"^\(([a-z])\)\s*(.*)$")  # "(a) Employees earn..."


def is_footer(line: str) -> bool:
    return any(p.match(line) for p in FOOTER_PATTERNS)


def table_rows(page: pymupdf.Page, table) -> list[list[str]]:
    """Read each cell's text from its area on the page.

    table.extract() drops the spaces between words in this PDF, so we don't use it.
    """
    rows = []
    for row in table.rows:
        cells = [" ".join(page.get_textbox(pymupdf.Rect(c)).split()) if c else "" for c in row.cells]
        rows.append(cells)
    return rows


def table_to_markdown(rows: list[list[str]]) -> str:
    rows = [row for row in rows if any(row)]  # drop empty (decorative) rows
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def read_items(pdf_path: Path) -> list[tuple[str, str]]:
    """Return the document as an ordered list of ("line", text) and ("table", markdown)."""
    items: list[tuple[str, str]] = []
    for page in pymupdf.open(pdf_path):
        tables = list(page.find_tables())
        table_boxes = [pymupdf.Rect(t.bbox) for t in tables]

        positioned: list[tuple[float, str, str]] = []
        for t, box in zip(tables, table_boxes):
            positioned.append((box.y0, "table", table_to_markdown(table_rows(page, t))))

        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            block = pymupdf.Rect(x0, y0, x1, y1)
            if any(block.intersects(box) for box in table_boxes):
                continue
            for line in text.splitlines():
                line = " ".join(line.split())
                if line and not is_footer(line):
                    positioned.append((y0, "line", line))

        # Stable sort keeps line order inside a block.
        positioned.sort(key=lambda item: item[0])
        items += [(kind, content) for _, kind, content in positioned]
    return items


def is_subheading(rest: str) -> bool:
    """'4.1 Annual Leave' is a heading; '2.1 This policy applies to ...' is a clause."""
    return len(rest.split()) <= 4 and not rest.endswith(".")


def build_sections(items: list[tuple[str, str]]) -> tuple[list[dict], dict]:
    title_lines: list[str] = []
    meta_table = ""
    sections: list[dict] = []

    top_id, top_heading = "", ""
    sub_id, sub_heading = "", ""
    current: dict | None = None

    def start(entry_id: str, parent: str, heading: str, path: str, text: str) -> dict:
        entry = {"id": entry_id, "parent": parent, "heading": heading, "path": path, "text": text}
        sections.append(entry)
        return entry

    for kind, content in items:
        if kind == "table":
            if not sections and not top_id:
                meta_table = content  # the metadata table at the top of page 1
            elif current is not None:
                current["text"] = (current["text"] + "\n\n" + content).strip()
            else:
                print(f"WARNING: table with no section to attach to:\n{content}")
            continue

        line = content
        if END_MARKER.match(line):
            continue

        if m := TOP_HEADING.match(line):
            top_id, top_heading = m.group(1), m.group(2)
            sub_id, sub_heading = "", ""
            # Sections without numbered clauses (1, 3, 7, 10) are a single entry.
            current = start(top_id, "", top_heading, f"{top_id}. {top_heading}", "")
            continue

        if m := NUMBERED.match(line):
            number, rest = m.group(1), m.group(2)
            top_path = f"{top_id}. {top_heading}"
            if is_subheading(rest):
                sub_id, sub_heading = number, rest
                current = start(number, top_id, rest, f"{top_path} > {number} {rest}", "")
            else:
                sub_id, sub_heading = "", ""
                current = start(number, top_id, top_heading, top_path, rest)
            continue

        if (m := LETTERED.match(line)) and sub_id:
            entry_id = f"{sub_id}({m.group(1)})"
            path = f"{top_id}. {top_heading} > {sub_id} {sub_heading}"
            current = start(entry_id, sub_id, sub_heading, path, m.group(2))
            continue

        if current is None:
            title_lines.append(line)  # title and disclaimer before section 1
        else:
            current["text"] = (current["text"] + " " + line).strip()

    # Drop heading-only entries ("4. Types of Leave", "4.1 Annual Leave") that only group clauses.
    sections = [s for s in sections if s["text"]]

    header = {
        "id": "header",
        "parent": "",
        "heading": "Policy details",
        "path": "Policy details",
        "text": "\n\n".join([" ".join(title_lines[1:]), meta_table]).strip(),
    }
    return [header] + sections, {"title": title_lines[0] if title_lines else ""}


def build_meta(sections: list[dict], title: str) -> dict:
    header_text = sections[0]["text"]

    def field(name: str) -> str:
        m = re.search(rf"\|\s*{name}\s*\|\s*([^|]+?)\s*\|", header_text)
        return m.group(1) if m else ""

    all_text = " ".join(s["text"] for s in sections)
    email = re.search(r"[\w.-]+@[\w.-]+\.\w+", all_text)
    return {
        "title": title,
        "policy_number": field("Policy number"),
        "version": field("Version"),
        "effective_date": field("Effective date"),
        "owner": field("Policy owner"),
        "applies_to": field("Applies to"),
        "hr_contact": email.group(0).rstrip(".") if email else "",
    }


def coverage_report(pdf_path: Path, policy: dict) -> None:
    """Compare the draft's words with the PDF's plain text, in both directions.

    Uses page.get_text() (independent of the table/section logic above), so a bug in
    that logic shows up here instead of being compared against itself.
    """
    lines = []
    for page in pymupdf.open(pdf_path):
        lines += [ln for ln in page.get_text().splitlines() if not is_footer(ln.strip())]
    source_text = " ".join(ln for ln in lines if not END_MARKER.match(ln.strip()))

    word = re.compile(r"[A-Za-z0-9@.:'’-]+")

    def words(text: str) -> set[str]:
        return {w.strip(".:-").lower() for w in word.findall(text) if w.strip(".:-")}

    draft_text = " ".join(
        [policy["meta"]["title"]] + [s["text"] for s in policy["sections"]]
        + [s["path"] for s in policy["sections"]]
    )
    source, draft = words(source_text), words(draft_text)
    # Clause labels like "(a)" and "2.1" move from the text into ids; that's expected.
    missing = {w for w in source - draft if not re.fullmatch(r"[a-z]|\d+\.\d+", w)}
    added = draft - source

    print(f"\nSections: {len(policy['sections'])}")
    print("IDs:", ", ".join(s["id"] for s in policy["sections"]))
    print("Words in PDF but not in draft:", sorted(missing) or "none")
    print("Words in draft but not in PDF:", sorted(added) or "none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/policy.draft.json"))
    args = parser.parse_args()

    items = read_items(args.pdf)
    sections, info = build_sections(items)
    policy = {"meta": build_meta(sections, info["title"]), "sections": sections}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Draft written to {args.out}. Verify it against the PDF before use.")
    coverage_report(args.pdf, policy)


if __name__ == "__main__":
    main()
