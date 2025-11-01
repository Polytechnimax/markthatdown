#!/usr/bin/env python3
"""
translate_md_tokens.py
Translate Markdown while preserving the original Markdown syntax and line structure.

- Does NOT render to HTML. It edits text spans in-place.
- Preserves: headings markers (#), list markers (-, *, +, 1.), blockquotes (>),
  tables (| cells |), fenced code blocks, inline code, links' URLs/images' URLs, spacing.
- Translates: visible text (including link text, headings text, list item text, paragraph text, table cell text).

Usage:
  python translate_md_tokens.py input.md output.md --src en --tgt de
Optional:
  --model MODEL_NAME           # "Helsinki-NLP/opus-mt-en-de", "facebook/m2m100_1.2B", "facebook/nllb-200-3.3B", etc.
  --device cpu|cuda            # default cpu
  --batch-size 32              # tune for throughput
"""

import argparse
import re
from typing import List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ---------- Regexes ----------
FENCE_OPEN_RE = re.compile(r'^(\s*)(`{3,}|~{3,})(.*)$')
FENCE_CLOSE_RE = re.compile(r'^(\s*)(`{3,}|~{3,})(.*)$')

HEADING_RE = re.compile(r'^(\s{0,3})(#{1,6})([ \t]+)(.*)$')
LIST_RE = re.compile(r'^(\s*)(?:([-+*])|(\d+[.)]))([ \t]+)(.*)$')
BLOCKQUOTE_RE = re.compile(r'^(\s*>+\s*)(.*)$')
TABLE_ALIGN_RE = re.compile(r'^\s*:?[-]{2,}?:?\s*$')

INLINE_CODE_RE = re.compile(r'`([^`]+)`')
LINK_RE = re.compile(r'(!?)\[(.*?)\]\((.*?)\)')

# ---------- Model selection ----------
M2M_LANG_MAP = {"en":"en","de":"de","fr":"fr","es":"es","it":"it","pt":"pt","nl":"nl","pl":"pl","ru":"ru","ja":"ja","zh":"zh"}
NLLB_LANG_MAP = {
    "en":"eng_Latn","de":"deu_Latn","fr":"fra_Latn","es":"spa_Latn","it":"ita_Latn","pt":"por_Latn",
    "nl":"nld_Latn","pl":"pol_Latn","ru":"rus_Cyrl","ja":"jpn_Jpan","zh":"zho_Hans","zh-Hans":"zho_Hans","zh-Hant":"zho_Hant"
}

def get_pipeline(src: str, tgt: str, device: str = "cpu", model_name: str = None, batch_size: int = 32):
    is_cuda = (device == "cuda" and torch.cuda.is_available())
    device_idx = 0 if is_cuda else -1
    candidate = model_name or f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    backend = "marian"
    if "m2m100" in (candidate.lower()):
        backend = "m2m"
    elif "nllb" in (candidate.lower()):
        backend = "nllb"
    try:
        tok = AutoTokenizer.from_pretrained(candidate)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(candidate)
        pipe = pipeline("translation", model=mdl, tokenizer=tok, device=device_idx, batch_size=batch_size)
        return pipe, (backend, candidate)
    except Exception:
        for fallback, bname in [("facebook/m2m100_1.2B","m2m"), ("facebook/m2m100_418M","m2m")]:
            try:
                tok = AutoTokenizer.from_pretrained(fallback)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(fallback)
                pipe = pipeline("translation", model=mdl, tokenizer=tok, device=device_idx, batch_size=batch_size)
                return pipe, (bname, fallback)
            except Exception:
                continue
        raise

def translate_texts(pipe, texts: List[str], src: str, tgt: str, backend_name: str) -> List[str]:
    if not texts:
        return []
    if backend_name == "m2m":
        tok = pipe.tokenizer
        tok.src_lang = M2M_LANG_MAP.get(src, src)
        tgt_code = M2M_LANG_MAP.get(tgt, tgt)
        if hasattr(tok, "get_lang_id"):
            forced_bos_token_id = tok.get_lang_id(tgt_code)
        else:
            forced_bos_token_id = tok.convert_tokens_to_ids(f"<<{tgt_code}>>")
        outs = pipe(texts, max_length=1024, clean_up_tokenization_spaces=True, forced_bos_token_id=forced_bos_token_id)
        return [o["translation_text"] for o in outs]
    if backend_name == "nllb":
        tok = pipe.tokenizer
        tok.src_lang = NLLB_LANG_MAP.get(src, src)
        tgt_code = NLLB_LANG_MAP.get(tgt, tgt)
        outs = pipe(texts, max_length=1024, clean_up_tokenization_spaces=True, tgt_lang=tgt_code)
        return [o["translation_text"] for o in outs]
    outs = pipe(texts, max_length=1024, clean_up_tokenization_spaces=True)
    return [o["translation_text"] for o in outs]

# ---------- Protectors ----------
def protect_inline_code(s: str) -> Tuple[str, List[str]]:
    codes = []
    def repl(m):
        idx = len(codes)
        codes.append(m.group(1))
        return f"@@CODE{idx}@@"
    return INLINE_CODE_RE.sub(repl, s), codes

def restore_inline_code(s: str, codes: List[str]) -> str:
    for i, code in enumerate(codes):
        s = s.replace(f"@@CODE{i}@@", f"`{code}`")
    return s

def protect_links(s: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    links = []
    def repl(m):
        bang, text, url = m.groups()
        idx = len(links)
        links.append((bang, text, url))
        return f"@@LINK{idx}@@"
    return LINK_RE.sub(repl, s), links

def restore_links(s: str, links: List[Tuple[str, str, str]], translated_texts: List[str]) -> str:
    for i, (bang, _orig, url) in enumerate(links):
        s = s.replace(f"@@LINK{i}@@", f"{bang}[{translated_texts[i]}]({url})")
    return s

def translate_span(pipe, span: str, src: str, tgt: str, backend_name: str) -> str:
    if not span.strip():
        return span
    # protect inline code & links so punctuation stays intact
    protected, codes = protect_inline_code(span)
    protected_links, links = protect_links(protected)
    main = translate_texts(pipe, [protected_links], src, tgt, backend_name)[0]
    if links:
        link_texts = [t for (_b,t,_u) in links]
        link_trans = translate_texts(pipe, link_texts, src, tgt, backend_name)
        main = restore_links(main, links, link_trans)
    main = restore_inline_code(main, codes)
    return main

def process_table_row(pipe, line: str, src: str, tgt: str, backend_name: str) -> str:
    # Split keeping pipes; translate non-align cells
    parts = line.split('|')
    # If alignment row, return as is
    def is_align_cell(cell: str) -> bool:
        return TABLE_ALIGN_RE.match(cell.strip()) is not None
    if all(is_align_cell(c) or c.strip()=='' for c in parts):
        return line
    # Translate cells individually (excluding leading/trailing empty due to leading/trailing pipes)
    new_parts = []
    for cell in parts:
        c = cell
        if not is_align_cell(c):
            c = translate_span(pipe, c, src, tgt, backend_name)
        new_parts.append(c)
    return '|'.join(new_parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pipe, (backend_name, model_used) = get_pipeline(args.src, args.tgt, device=args.device, model_name=args.model, batch_size=args.batch_size)

    out_lines = []
    in_fence = False
    fence_tick = None

    buffer_for_batch: List[Tuple[int, str]] = []  # (index in out_lines, text_to_translate)

    def flush_batch():
        if not buffer_for_batch:
            return
        # translate in one go
        texts = [t for (_i, t) in buffer_for_batch]
        trans = translate_texts(pipe, texts, args.src, args.tgt, backend_name)
        for (i, _), t in zip(buffer_for_batch, trans):
            out_lines[i] = t
        buffer_for_batch.clear()

    for line in lines:
        # Detect fences (``` or ~~~)
        if not in_fence:
            m = FENCE_OPEN_RE.match(line)
            if m:
                in_fence = True
                fence_tick = m.group(2)[0]
                out_lines.append(line)
                continue
        else:
            # inside fence
            out_lines.append(line)
            if FENCE_CLOSE_RE.match(line) and line.strip().startswith(fence_tick*3):
                in_fence = False
                fence_tick = None
            continue

        # Headings
        m = HEADING_RE.match(line)
        if m:
            prefix = m.group(1) + m.group(2) + m.group(3)
            text = m.group(4)
            # translate span but keep leading/trailing whitespace of the text as-is
            lead = len(text) - len(text.lstrip(" \t"))
            trail = len(text) - len(text.rstrip(" \t"))
            left = text[:lead]
            right = "" if trail == 0 else text[-trail:]
            translated = translate_span(pipe, text.strip(), args.src, args.tgt, backend_name)
            out_lines.append(prefix + left + translated + right + "\n" if not line.endswith("\n") else prefix + left + translated + right + "")
            continue

        # Blockquotes
        m = BLOCKQUOTE_RE.match(line)
        if m:
            marker = m.group(1)
            rest = m.group(2)
            # We keep exact spacing after '>'
            i = len(out_lines)
            out_lines.append("")  # placeholder
            buffer_for_batch.append((i, marker + translate_span(pipe, rest, args.src, args.tgt, backend_name)))
            continue

        # Lists
        m = LIST_RE.match(line)
        if m:
            prefix = m.group(1) + (m.group(2) or m.group(3)) + m.group(4)
            text = m.group(5)
            i = len(out_lines)
            out_lines.append("")
            buffer_for_batch.append((i, prefix + translate_span(pipe, text, args.src, args.tgt, backend_name)))
            continue

        # Tables (has pipes and not an alignment-only row)
        if '|' in line and not line.lstrip().startswith(('<','```','~~~')):
            new_line = process_table_row(pipe, line, args.src, args.tgt, backend_name)
            out_lines.append(new_line)
            continue

        # Blank line
        if line.strip() == "":
            out_lines.append(line)
            continue

        # Default: translate the entire line as a span (paragraph text)
        i = len(out_lines)
        out_lines.append("")
        buffer_for_batch.append((i, translate_span(pipe, line, args.src, args.tgt, backend_name)))

    # Flush any buffered batch items
    flush_batch()

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"Done. Backend: {backend_name}, model: {model_used}")
if __name__ == "__main__":
    main()