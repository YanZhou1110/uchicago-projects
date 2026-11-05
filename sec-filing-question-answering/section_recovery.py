"""
================================================================================
 Section Recovery Module — Fix Parser Bug in SEC 10-K Corpus
================================================================================
The corpus has a known issue: filenames don't reliably match content. E.g.
AAPL_2022_Item7.txt actually contains Item 1A content, while the real Item 7
sits inside AAPL_2022_Item8.txt. This module re-parses each raw file using
regex on actual SEC section headers, recovering true section boundaries.

Drop-in replacement: replace `find_section_file()` + naive read with
`load_recovered_section()` in your Full-Doc Challenger.
================================================================================
"""

import os
import re
import glob

# Robust regex patterns for SEC 10-K section headers
# Match both "Item X." and "ITEM X." with various whitespace
SECTION_PATTERNS = [
    ('Item1',  r'(?:^|\n)\s*(?:Item|ITEM)\s+1\.?\s+(?:Business|BUSINESS)\b'),
    ('Item1A', r'(?:^|\n)\s*(?:Item|ITEM)\s+1A\.?\s+(?:Risk Factors|RISK FACTORS)'),
    ('Item2',  r'(?:^|\n)\s*(?:Item|ITEM)\s+2\.?\s+(?:Properties|PROPERTIES)\b'),
    ('Item3',  r'(?:^|\n)\s*(?:Item|ITEM)\s+3\.?\s+(?:Legal Proceedings|LEGAL PROCEEDINGS)'),
    ('Item5',  r'(?:^|\n)\s*(?:Item|ITEM)\s+5\.?\s+(?:Market|MARKET)'),
    ('Item6',  r'(?:^|\n)\s*(?:Item|ITEM)\s+6\.?\s+(?:\[Reserved\]|Reserved|Selected|SELECTED)'),
    ('Item7',  r'(?:^|\n)\s*(?:Item|ITEM)\s+7\.?\s+(?:Management|MANAGEMENT)'),
    ('Item7A', r'(?:^|\n)\s*(?:Item|ITEM)\s+7A\.?\s+(?:Quantitative|QUANTITATIVE)'),
    ('Item8',  r'(?:^|\n)\s*(?:Item|ITEM)\s+8\.?\s+(?:Financial|FINANCIAL)'),
    ('Item9',  r'(?:^|\n)\s*(?:Item|ITEM)\s+9\.?\s+(?:Changes|CHANGES|Controls|CONTROLS)'),
]


def split_into_sections(raw_text):
    """
    Split raw 10-K text into a dict of {section_name: content_text}.
    Only sections with detectable headers are returned. Pre-header
    content (often the misallocated Item 1A body) is keyed as
    '_preamble'.
    """
    positions = []
    for name, pat in SECTION_PATTERNS:
        for m in re.finditer(pat, raw_text):
            positions.append((m.start(), name))
    positions.sort()

    sections = {}
    if not positions:
        sections['_preamble'] = raw_text
        return sections

    # Pre-header chunk (often misallocated content)
    if positions[0][0] > 100:
        sections['_preamble'] = raw_text[:positions[0][0]]

    for i, (pos, name) in enumerate(positions):
        end = positions[i+1][0] if i + 1 < len(positions) else len(raw_text)
        if name not in sections:  # keep first occurrence
            sections[name] = raw_text[pos:end]

    return sections


def load_recovered_section(ticker, target_section, fiscal_year,
                           corpus_path, exclude_fn=None):
    """
    Search ALL files for this ticker+year, parse each, and return the
    target section if found in ANY of them.

    Args:
        ticker:          'AAPL', 'MSFT', etc.
        target_section:  'Item1', 'Item1A', 'Item7', 'Item8' (no space)
        fiscal_year:     '2022' (string)
        corpus_path:     path to sec_corpus_clean/
        exclude_fn:      optional should_exclude(filename) function

    Returns:
        Section text (string), or None if not found in any file.
    """
    pattern = os.path.join(corpus_path, f"{ticker}_*_{fiscal_year}_Item*.txt")
    all_files = glob.glob(pattern)
    if exclude_fn is not None:
        all_files = [f for f in all_files if not exclude_fn(os.path.basename(f))]

    # First pass: check files whose name matches target_section (fast path)
    name_match_files = [f for f in all_files
                        if f.endswith(f"_{target_section}.txt")]
    other_files = [f for f in all_files if f not in name_match_files]
    ordered = name_match_files + other_files

    for filepath in ordered:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = f.read()
        except Exception:
            continue
        sections = split_into_sections(raw)
        if target_section in sections:
            return sections[target_section]

    # Fallback: if target is Item1A and we only found _preamble in the
    # named file, return that (the preamble is usually misallocated Item 1A)
    if target_section == 'Item1A':
        for filepath in ordered:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw = f.read()
            except Exception:
                continue
            sections = split_into_sections(raw)
            if '_preamble' in sections and len(sections['_preamble']) > 5000:
                # Heuristic: long preamble in a non-Item1 file is likely Item 1A content
                return sections['_preamble']

    return None


def assemble_recovered_full_doc(ticker, question, fiscal_year, corpus_path,
                                 max_words=8000, exclude_fn=None):
    """
    Drop-in replacement for `assemble_full_doc_prompt`. Uses content-aware
    section recovery instead of trusting filenames.
    """
    all_sections = []
    for section_raw in ['Item1', 'Item1A', 'Item7', 'Item8']:
        content = load_recovered_section(
            ticker, section_raw, fiscal_year, corpus_path, exclude_fn
        )
        if content is None:
            continue
        all_sections.append(f"=== {section_raw} ===\n{content}")

    if not all_sections:
        return f"ERROR: No section content recovered for {ticker} FY{fiscal_year}."

    full_doc = "\n\n".join(all_sections)
    words = full_doc.split()
    if len(words) > max_words:
        full_doc = " ".join(words[:max_words]) + "\n...[TRUNCATED]"

    prompt = f"""You are an expert financial analyst. Answer the user's question using ONLY the provided text.

Company Ticker: {ticker}
Fiscal Year: {fiscal_year}

--- 10-K CONTENT (multiple sections concatenated) ---
{full_doc}
--- END CONTENT ---

Question: {question}
Answer:"""
    return prompt


# =========================== USAGE IN COLAB ==================================
# 1. Save this file to [local path omitted]
# 2. In your notebook, add a cell:
#
#       import sys
#       sys.path.insert(0, 'data/SEC_NLP')
#       from section_recovery import assemble_recovered_full_doc
#
# 3. Replace `assemble_full_doc_prompt(...)` calls with:
#
#       challenger_prompt = assemble_recovered_full_doc(
#           ticker, question, fiscal_year=fy,
#           corpus_path=G_DRIVE_CORPUS_PATH,
#           max_words=MAX_FULL_DOC_WORDS,
#           exclude_fn=should_exclude,
#       )
#
# 4. Re-run Section 9 (batch runner).
#
# Expected effect:
#   - Challenger now sees REAL Item 7 (MD&A) + REAL Item 8 (financials)
#   - Factual_lookup questions about $394,328 etc. should now have answer in context
#   - EM may rise from 0.000 to 0.25-0.75
#   - Faithfulness should recover from 0.375 back to ~0.6
#   - Champion (RAG) remains unchanged - already section-agnostic
# =============================================================================
