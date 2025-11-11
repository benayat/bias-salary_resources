"""Read an HTML table into pandas safely and print a small preview.

This script fixes two issues:
- `pd.read_html` returns a list of DataFrames; pick the requested table index.
- For very large tables there's an optional streaming fallback which uses
  lxml.iterparse (only used when --stream is passed and lxml is installed).

Usage examples:
  python3 extract_table_from_html.py salaries.html --rows 10
  python3 extract_table_from_html.py salaries.html --table-index 0 --out csv --out-path data.csv
  python3 extract_table_from_html.py salaries.html --stream --chunk-size 50000
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Iterator, List, Optional

import pandas as pd


def read_with_pandas(path: str, table_index: int = 0) -> pd.DataFrame:
  """Use pandas.read_html and return the selected table (default first).

  read_html returns a list of DataFrames; this helper chooses the requested
  table and raises a helpful error if the index is out of range.
  """
  tables = pd.read_html(path)
  if not tables:
    raise ValueError(f"No tables found in {path!r}")
  if table_index < 0 or table_index >= len(tables):
    raise IndexError(f"table_index {table_index} out of range (0..{len(tables)-1})")
  return tables[table_index]


def stream_table(path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
  """Stream the first <table> found in the HTML and yield DataFrame chunks.

  This uses lxml.etree.iterparse and keeps memory bounded by clearing parsed
  elements. If lxml is not available an ImportError will be raised.
  """
  try:
    from lxml import etree
  except Exception as e:  # ImportError or similar
    raise ImportError("lxml is required for streaming parsing. Install with: pip install lxml") from e

  parser = etree.HTMLParser()
  # We'll look for the first <table> and parse its <tr> rows.
  header: Optional[List[str]] = None
  rows: List[List[str]] = []

  # iterparse the file and react to start/end events
  context = etree.iterparse(path, events=("start", "end"), parser=parser)
  inside_table = False

  for event, elem in context:
    tag = elem.tag.lower() if isinstance(elem.tag, str) else None
    if event == "start" and tag == "table" and not inside_table:
      inside_table = True
      header = None
      rows = []

    if not inside_table:
      # not yet inside the first table, continue
      continue

    # When we hit the end of a tr inside the target table, extract its cells
    if event == "end" and tag == "tr":
      # extract text content of th/td children
      cells: List[str] = []
      for child in elem:
        if not isinstance(child.tag, str):
          continue
        child_tag = child.tag.lower()
        if child_tag in ("td", "th"):
          text = "".join(child.itertext()).strip()
          cells.append(text)

      # If header is not set and this tr has th cells, use them; otherwise
      # assume the first row is header if header is None.
      if header is None:
        header = cells
      else:
        # align row length with header
        if len(cells) < len(header):
          cells += [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
          cells = cells[: len(header)]
        rows.append(cells)

      # Clear the element to keep memory usage low
      elem.clear()

      if len(rows) >= chunk_size:
        yield pd.DataFrame(rows, columns=header)
        rows = []

    # If we hit the end of the table, yield any remaining rows and stop
    if event == "end" and tag == "table" and inside_table:
      if rows:
        yield pd.DataFrame(rows, columns=header)
      # Break out after the first table is processed
      break


def main(argv: Optional[List[str]] = None) -> int:
  parser = argparse.ArgumentParser(description="Extract HTML table(s) to pandas and preview/save them")
  parser.add_argument("--path", default='salaries.html' , help="Path to the HTML file containing the table(s)")
  parser.add_argument("--table-index", type=int, default=0, help="Which table to select (default 0)")
  parser.add_argument("--rows", type=int, default=5, help="How many rows to show from the table head")
  parser.add_argument("--out", choices=("none", "csv"), default="none", help="Optional output format")
  parser.add_argument("--out-path", default="table.csv", help="Output path when --out=csv")
  parser.add_argument("--stream", action="store_true", help="Use streaming parser to avoid loading whole table (requires lxml)")
  parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size (rows) when streaming")

  args = parser.parse_args(argv)

  try:
    if args.stream:
      # Streaming: iterate over chunks and show the head of the first chunk
      gen = stream_table(args.path, chunk_size=args.chunk_size)
      first_chunk = None
      for df_chunk in gen:
        if first_chunk is None:
          first_chunk = df_chunk
          break

      if first_chunk is None:
        print(f"No table rows found in {args.path}")
        return 1

      print(first_chunk.head(args.rows).to_string(index=False))
      if args.out == "csv":
        first_chunk.to_csv(args.out_path, index=False)
    else:
      df = read_with_pandas(args.path, table_index=args.table_index)
      print(df.head(args.rows).to_string(index=False))
      if args.out == "csv":
        df.to_csv(args.out_path, index=False)

  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 2

  return 0


if __name__ == "__main__":
  raise SystemExit(main())