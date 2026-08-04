"""
One-off cleanup for raw SEC EDGAR full-submission text files.

A file downloaded from EDGAR's full-text archive (e.g. an .txt submission)
bundles the SGML header, the actual 10-K as inline-XBRL HTML, and every
exhibit/XBRL taxonomy file into one giant document. Ingesting that raw
blob produces tens of thousands of tiny, mostly-markup chunks and blows
through the embedding API's rate limit.

This script pulls out just the first <DOCUMENT> block (the actual 10-K),
strips HTML/XBRL tags and hidden facts, and writes plain readable text.

Usage:
    python scripts/clean_sec_filing.py <input.txt> <output.txt>
"""
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup


def extract_first_document_text(raw: str) -> str:
    doc_start = raw.index("<DOCUMENT>")
    doc_end = raw.index("</DOCUMENT>", doc_start)
    document = raw[doc_start:doc_end]

    text_start = document.index("<TEXT>") + len("<TEXT>")
    text_end = document.index("</TEXT>", text_start)
    return document[text_start:text_end]


# Inline XBRL wraps every individual number/word in its own tag, so a
# plain get_text(separator="\n") shatters table rows and sentences across
# many single-token lines. Only break lines at real block boundaries
# (rows, paragraphs, headings) and keep everything inside one row/paragraph
# space-joined, so "$" and "134,711" stay on the same line.
BLOCK_TAGS = ["tr", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]


def html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Non-rendered inline-XBRL facts (continuation text, hidden fields)
    for tag in soup.select('[style*="display:none"], [style*="display: none"], ix\\:hidden'):
        tag.decompose()

    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after("\n")

    text = soup.get_text(separator=" ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/clean_sec_filing.py <input.txt> <output.txt>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    raw = input_path.read_text(encoding="utf-8", errors="ignore")
    print(f"Input: {len(raw):,} chars")

    document_html = extract_first_document_text(raw)
    print(f"Extracted <DOCUMENT> text section: {len(document_html):,} chars")

    clean_text = html_to_clean_text(document_html)
    print(f"Clean text: {len(clean_text):,} chars")

    output_path.write_text(clean_text, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
