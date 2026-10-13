import os
import sys
import base64
import fitz  # PyMuPDF
from pathlib import Path
from groq import Groq
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.rule import Rule
from collections import Counter
from PIL import Image
import io
import re
import json
import time
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
API_KEY   = os.environ.get("GROQ_API_KEY")
MODEL_ID  = "qwen/qwen3.6-27b"
MAX_BATCH = 3          # images per API call (Groq limit)
MAX_HIST  = 6          # Q&A pairs to keep in memory

console = Console()


# ─────────────────────────────────────────
#  LATEX → UNICODE CONVERTER
# ─────────────────────────────────────────
def _strip_latex(text: str) -> str:
    """Convert LaTeX math notation to plain Unicode. Applied to ALL model output."""
    replacements = [
        # Core operators
        (r'\\int\b',          '∫'),
        (r'\\iint\b',         '∬'),
        (r'\\iiint\b',        '∭'),
        (r'\\oint\b',         '∮'),
        (r'\\sum\b',          '∑'),
        (r'\\prod\b',         '∏'),
        (r'\\partial\b',      '∂'),
        (r'\\nabla\b',        '∇'),
        (r'\\infty\b',        '∞'),
        (r'\\sqrt\{([^}]+)\}','√(\\1)'),
        (r'\\sqrt\b',         '√'),
        (r'\\pm\b',           '±'),
        (r'\\times\b',        '×'),
        (r'\\div\b',          '÷'),
        (r'\\cdot\b',         '·'),
        (r'\\leq\b',          '≤'),
        (r'\\geq\b',          '≥'),
        (r'\\neq\b',          '≠'),
        (r'\\approx\b',       '≈'),
        (r'\\le\b',           '≤'),
        (r'\\ge\b',           '≥'),
        # Greek letters
        (r'\\alpha\b',        'α'),
        (r'\\beta\b',         'β'),
        (r'\\gamma\b',        'γ'),
        (r'\\delta\b',        'δ'),
        (r'\\epsilon\b',      'ε'),
        (r'\\theta\b',        'θ'),
        (r'\\lambda\b',       'λ'),
        (r'\\mu\b',           'μ'),
        (r'\\pi\b',           'π'),
        (r'\\sigma\b',        'σ'),
        (r'\\phi\b',          'φ'),
        (r'\\omega\b',        'ω'),
        (r'\\Sigma\b',        'Σ'),
        (r'\\Delta\b',        'Δ'),
        (r'\\Pi\b',           'Π'),
        # Trig / log
        (r'\\ln\b',           'ln'),
        (r'\\log\b',          'log'),
        (r'\\sin\b',          'sin'),
        (r'\\cos\b',          'cos'),
        (r'\\tan\b',          'tan'),
        (r'\\sec\b',          'sec'),
        (r'\\csc\b',          'csc'),
        (r'\\cot\b',          'cot'),
        (r'\\arcsin\b',       'arcsin'),
        (r'\\arccos\b',       'arccos'),
        (r'\\arctan\b',       'arctan'),
        (r'\\exp\b',          'exp'),
        (r'\\lim\b',          'lim'),
        (r'\\max\b',          'max'),
        (r'\\min\b',          'min'),
        # Fractions: \frac{a}{b} → (a/b)
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)'),
        # Superscripts: ^{n} → ^n
        (r'\^\{([^}]+)\}',    r'^\1'),
        # Subscripts: _{n} → _n
        (r'_\{([^}]+)\}',     r'_\1'),
        # Brackets
        (r'\\left[\(\[\{]',   '('),
        (r'\\right[\)\]\}]',  ')'),
        (r'\\left\|',         '|'),
        (r'\\right\|',        '|'),
        # Misc
        (r'\\boxed\{([^}]+)\}', r'\1'),
        (r'\\text\{([^}]+)\}',  r'\1'),
        (r'\\mathrm\{([^}]+)\}',r'\1'),
        (r'\\mathbf\{([^}]+)\}',r'\1'),
        (r'\\,',              ' '),
        (r'\\;',              ' '),
        (r'\\!',              ''),
        (r'\\quad\b',         '  '),
        (r'\\qquad\b',        '    '),
        (r'\\\\',             '\n'),
        (r'\\\s',             ' '),
    ]

    result = text
    for pattern, repl in replacements:
        result = re.sub(pattern, repl, result)

    # Strip $ delimiters
    result = re.sub(r'\$\$(.+?)\$\$', r'\n\1\n', result, flags=re.DOTALL)
    result = re.sub(r'\$([^$\n]+)\$', r'\1', result)

    # Clean up leftover braces
    result = re.sub(r'(?<![a-zA-Z])\{([^{}]+)\}(?![a-zA-Z])', r'\1', result)

    # Superscript digits: x^2 → x²
    sup_map = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵',
               '6':'⁶','7':'⁷','8':'⁸','9':'⁹','n':'ⁿ'}
    def replace_sup(m):
        return ''.join(sup_map.get(c, c) for c in m.group(1))
    result = re.sub(r'\^([0-9n])\b', replace_sup, result)

    return result


def _api_call(client, **kwargs):
    """Wrapper with retry on rate limits."""
    max_retries = 4
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'rate_limit' in err_str.lower():
                wait = 90
                m = re.search(r'try again in\s+([\d.]+)([smh])', err_str, re.IGNORECASE)
                if m:
                    val, unit = float(m.group(1)), m.group(2).lower()
                    wait = int(val * {'s': 1, 'm': 60, 'h': 3600}.get(unit, 1)) + 5
                if attempt < max_retries - 1:
                    console.print(f"\n[yellow]  Rate limit hit — waiting {wait}s before retry ({attempt+1}/{max_retries-1})...[/yellow]")
                    for remaining in range(wait, 0, -1):
                        console.print(f"[dim]  Retrying in {remaining}s...[/dim]", end='\r')
                        time.sleep(1)
                    console.print(" " * 40, end='\r')
                else:
                    raise RuntimeError(
                        f"Rate limit exceeded and retries exhausted.\n"
                        f"You've used up your daily token quota on Groq's free tier (500k tokens/day).\n"
                        f"Wait until midnight UTC for the quota to reset, or upgrade at https://console.groq.com/settings/billing"
                    )
            else:
                raise


# ─────────────────────────────────────────
#  CONTENT TYPE DETECTION
# ─────────────────────────────────────────
MATH_SYMBOLS = re.compile(
    r'[∫∬∭∮√∛∜∑∏∂∇∞≈≠≤≥±×÷αβγδεζηθλμπσφψω]'
    r'|\\(int|frac|sqrt|sum|prod|lim|infty|partial|nabla|alpha|beta|gamma|delta|theta|pi|sigma)\b'
    r'|\b(integrate|derivative|differentiate|solve for|evaluate|simplify|factor|expand)\b'
    r'|\d+\s*[\+\-\*\/\^]\s*\d+'
    r'|=\s*[\d\+\-\(\)\[\]]'
    r'|\b(dx|dy|dz|dt|d[a-z])\b'
    r'|x\^[0-9n]|[a-z]\^[0-9]',
    re.IGNORECASE
)

NON_MATH_QUESTION = re.compile(
    r'^\s*(which|what|where|when|how|who|why|define|explain|describe|list|name|'
    r'identify|state|compare|contrast|discuss|outline|give|write|mention)\b',
    re.IGNORECASE
)

SOLVE_PATTERN = re.compile(
    r'^\s*(solve|calculate|compute|find|evaluate|integrate|differentiate|simplify|factor)\b',
    re.IGNORECASE
)

def classify_content(text: str) -> str:
    """Returns 'math', 'text', or 'mixed'"""
    if SOLVE_PATTERN.match(text.strip()):
        return 'math'
    has_math = bool(MATH_SYMBOLS.search(text))
    has_non_math_q = bool(NON_MATH_QUESTION.search(text))
    if has_math and not has_non_math_q:
        return 'math'
    if has_math:
        return 'mixed'
    return 'text'


# ─────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────
BASE_RULES = """You are an expert document analyst with strong reasoning skills.

CORE RULES:
1. Always cite page numbers: "According to page X..." or "(Page X)"
2. Only reference pages currently visible to you — never cite pages outside the valid range
3. Never fabricate information not present in the document
4. Give your best answer from the visible pages — do NOT end responses asking for more pages
5. Think step by step before answering

CONTENT DIFFERENTIATION:
- Math content: equations, integrals, derivatives, algebra, geometry, statistics
- Text content: paragraphs, explanations, definitions, descriptions
- Treat them separately — never confuse an essay question with a math problem

"""

MATH_FORMAT_RULES = """
MATH FORMATTING — ABSOLUTE RULE, NO EXCEPTIONS EVER:

BEFORE writing any math, ask yourself: "Am I about to write a $ sign or backslash?"
If yes — STOP and rewrite using Unicode instead.

BANNED (will never appear in your response):
  $ $$ \\int \\frac \\sqrt \\cdot \\left \\right \\boxed \\ln \\sin \\cos
  Any LaTeX command starting with backslash
  Any math wrapped in dollar signs

REQUIRED Unicode symbols:
  Integrals:   ∫ f(x) dx
  Fractions:   (a/b)
  Exponents:   x² x³ xⁿ  or  x^2 x^3
  Roots:       √x  √(x+1)
  Absolute:    |x+1|
  Multiply:    x·y
  Greek:       α β γ δ θ π λ μ σ φ
  Other:       ∂ ∇ ∑ ∏ ∞ ≈ ≠ ≤ ≥ ± × ÷

CORRECT — write exactly like this:
  ∫ (x+1)/(x²+2x+1) dx
  ∫ (1/(x+1)) dx = ln|x+1| + C
  f'(x) = 3x² + 2x
  √(x² + y²)

WRONG — never write like this:
  $\\int \\frac{x+1}{x^2+2x+1} dx$    NO
  $\\ln|x+1| + C$                      NO
  $f'(x) = 3x^2 + 2x$                  NO

MATH SOLVING FORMAT:
Problem: [state using Unicode math]

Method: [what technique to use and why]

Work:
  Step 1 — [label]
    [Unicode math only]
  Step 2 — [label]
    [continue]

Result: [final answer in Unicode, no $ signs]

"""

TEXT_ANALYSIS_RULES = """
TEXT ANALYSIS GUIDELINES:
- For summaries: organize by theme, not page by page
- For questions: answer directly, then cite source page
- For definitions: give the exact definition from the document, then explain
- For diagrams/tables/graphs: describe what they show and what it means
- For data: extract key numbers and what they indicate
- Use markdown headers and bullets to organize complex answers

"""

def get_system_prompt(doc_type: str) -> str:
    base = BASE_RULES
    if doc_type == 'math':
        return base + MATH_FORMAT_RULES + "PRIMARY MODE: You are solving mathematical problems. Show all work.\n"
    elif doc_type == 'text':
        return base + TEXT_ANALYSIS_RULES + "PRIMARY MODE: You are analyzing textual/educational content.\n"
    else:
        return base + MATH_FORMAT_RULES + TEXT_ANALYSIS_RULES + (
            "PRIMARY MODE: This document has both math and text. "
            "Detect which type each question is about and respond accordingly.\n"
        )


# ─────────────────────────────────────────
#  PAGE CHUNK MANAGER
# ─────────────────────────────────────────
class PageChunkManager:
    def __init__(self, context_data: List[Dict], total_pages: int):
        self.context_data = context_data
        self.total_pages  = total_pages

    def get_pages(self, page_numbers: List[int]) -> List[Dict]:
        result = []
        for p in page_numbers:
            if 1 <= p <= self.total_pages:
                result.append(self.context_data[p - 1])
        return result[:MAX_BATCH]

    def get_chunk(self, start: int, size: int = MAX_BATCH) -> List[int]:
        end = min(start + size - 1, self.total_pages)
        return list(range(start, end + 1))

    def all_chunks(self, size: int = MAX_BATCH) -> List[List[int]]:
        chunks = []
        for s in range(1, self.total_pages + 1, size):
            chunks.append(self.get_chunk(s, size))
        return chunks


# ─────────────────────────────────────────
#  DOCUMENT INDEX
# ─────────────────────────────────────────
class DocumentIndex:
    def __init__(self):
        self.pages: Dict[int, Dict] = {}
        self.topics: Dict[str, List[int]] = {}
        self.math_problems: List[Dict] = []
        self.sections: List[Dict] = []
        self.doc_type: str = 'text'
        self.has_images: List[int] = []
        self.has_tables: List[int] = []
        self.has_graphs: List[int] = []
        # FIX: direct map from problem number → page number
        self.problem_number_map: Dict[int, int] = {}

    def add_page(self, page_num: int, data: Dict):
        if not (1 <= page_num <= 9999):
            return

        self.pages[page_num] = data

        # Index topics
        for topic in data.get('topics', []):
            t = topic.strip()
            if not t or t.lower() in ('content', 'unknown', ''):
                continue
            canonical = t.title()
            existing_key = next(
                (k for k in self.topics if k.lower() == t.lower()), None
            )
            if existing_key:
                if page_num not in self.topics[existing_key]:
                    self.topics[existing_key].append(page_num)
            else:
                self.topics[canonical] = []
                if page_num not in self.topics[canonical]:
                    self.topics[canonical].append(page_num)

        # FIX: handle both dict-style {number, text} and plain string problems
        for prob in data.get('math_problems', []):
            if isinstance(prob, dict):
                num  = prob.get('number')
                text = prob.get('text', '')
                if isinstance(text, str):
                    text = _strip_latex(text)
                # Register in number map
                if num is not None:
                    try:
                        self.problem_number_map[int(num)] = page_num
                    except (ValueError, TypeError):
                        pass
                if text and self._is_real_math(text):
                    self.math_problems.append({
                        'page': page_num,
                        'description': text,
                        'number': num
                    })
            elif isinstance(prob, str):
                text = _strip_latex(prob)
                # Try to extract a leading problem number from the text itself
                # e.g. "108. ∫ sin⁶x cos⁵x dx"
                m = re.match(r'^(\d+)[.\)]\s*(.+)', text)
                if m:
                    try:
                        extracted_num = int(m.group(1))
                        self.problem_number_map[extracted_num] = page_num
                    except ValueError:
                        pass
                if self._is_real_math(text):
                    self.math_problems.append({
                        'page': page_num,
                        'description': text,
                        'number': None
                    })

        # Index sections
        section = data.get('section', '')
        if section and section.lower() not in ('unknown', '', 'null', 'none'):
            existing_count = sum(
                1 for s in self.sections if s['title'].lower() == section.lower()
            )
            if existing_count < 2:
                self.sections.append({'page': page_num, 'title': section})

        if data.get('has_images'): self.has_images.append(page_num)
        if data.get('has_tables'): self.has_tables.append(page_num)
        if data.get('has_graphs'): self.has_graphs.append(page_num)

    def _is_real_math(self, text: str) -> bool:
        if NON_MATH_QUESTION.match(text.strip()):
            return False
        return bool(MATH_SYMBOLS.search(text))

    def deduplicate_sections(self):
        title_counts = Counter(s['title'].lower() for s in self.sections)
        seen = set()
        clean = []
        for s in sorted(self.sections, key=lambda x: x['page']):
            title_lower = s['title'].lower()
            if title_counts[title_lower] == 1:
                clean.append(s)
            elif title_lower not in seen:
                clean.append(s)
                seen.add(title_lower)
        self.sections = clean

    def determine_doc_type(self):
        real_math   = len(self.math_problems)
        total_pages = len(self.pages)
        if real_math == 0:
            self.doc_type = 'text'
        elif real_math >= total_pages * 2:
            self.doc_type = 'math'
        else:
            self.doc_type = 'mixed'
        return self.doc_type

    def find_pages_for_topic(self, query: str) -> List[int]:
        query_words = set(query.lower().split())
        scores: Dict[int, int] = {}
        for topic, pages in self.topics.items():
            overlap = len(query_words & set(topic.lower().split()))
            if overlap > 0:
                for p in pages:
                    scores[p] = scores.get(p, 0) + overlap
        for page_num, data in self.pages.items():
            summary  = data.get('summary', '').lower()
            concepts = ' '.join(data.get('key_concepts', [])).lower()
            combined = summary + ' ' + concepts
            for word in query_words:
                if len(word) > 3 and word in combined:
                    scores[page_num] = scores.get(page_num, 0) + 1
        return sorted(scores, key=scores.get, reverse=True)[:MAX_BATCH]


# ─────────────────────────────────────────
#  MAIN ASSISTANT
# ─────────────────────────────────────────
class DocumentAssistant:
    def __init__(self):
        if not API_KEY:
            console.print("[bold red]Error: GROQ_API_KEY not set.[/bold red]")
            console.print("[yellow]Run: $env:GROQ_API_KEY='your-key-here'[/yellow]")
            sys.exit(1)

        self.client       = Groq(api_key=API_KEY)
        self.context_data : List[Dict] = []
        self.doc_name     : str = ""
        self.total_pages  : int = 0
        self.index        = DocumentIndex()
        self.chunk_mgr    : Optional[PageChunkManager] = None
        self.messages     : List[Dict] = []
        self.full_summary : Optional[str] = None

    # ── FILE PROCESSING ──────────────────────────────────────────────────────

    def load_file(self, file_path: str) -> bool:
        path = Path(file_path.strip('"').strip("'"))
        if not path.exists():
            console.print(f"[bold red]File not found: {path}[/bold red]")
            return False

        self.doc_name     = path.name
        self.context_data = []

        try:
            suffix = path.suffix.lower()
            if suffix == '.pdf':
                return self._load_pdf(path)
            elif suffix in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'):
                return self._load_image(path)
            else:
                console.print(f"[bold red]Unsupported format: {suffix}[/bold red]")
                return False
        except Exception as e:
            console.print(f"[bold red]Error loading file: {e}[/bold red]")
            return False

    def _load_pdf(self, path: Path) -> bool:
        doc = fitz.open(path)
        self.total_pages = len(doc)
        console.print(f"\n[bold cyan]  Document:[/bold cyan] {self.doc_name}")
        console.print(f"[bold cyan]  Pages:[/bold cyan]    {self.total_pages}")

        scale   = 2.0 if self.total_pages <= 30 else 1.5
        quality = 85  if self.total_pages <= 20 else 70

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                console=console
            ) as prog:
                task = prog.add_task("[cyan]Rendering pages...", total=self.total_pages)
                for i in range(self.total_pages):
                    page     = doc.load_page(i)
                    pix      = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
                    img_data = pix.tobytes("jpeg", jpg_quality=quality)
                    self.context_data.append({
                        "page":  i + 1,
                        "image": base64.b64encode(img_data).decode()
                    })
                    prog.update(task, advance=1)
        finally:
            doc.close()

        console.print(f"[green]  Rendered {self.total_pages} pages successfully[/green]")
        return True

    def _load_image(self, path: Path) -> bool:
        self.total_pages = 1
        img = Image.open(path)
        if max(img.size) > 2048:
            ratio = 2048 / max(img.size)
            img   = img.resize(tuple(int(d * ratio) for d in img.size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=85)
        self.context_data.append({
            "page":  1,
            "image": base64.b64encode(buf.getvalue()).decode()
        })
        console.print(f"[green]  Image loaded successfully[/green]")
        return True

    # ── INDEXING ──────────────────────────────────────────────────────────────

    def build_index(self):
        self.chunk_mgr = PageChunkManager(self.context_data, self.total_pages)
        console.print()
        _print_rule("Analyzing Document")

        all_chunks = self.chunk_mgr.all_chunks()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=console
        ) as prog:
            task = prog.add_task(
                f"[cyan]Indexing {self.total_pages} pages...",
                total=len(all_chunks)
            )
            for chunk in all_chunks:
                self._analyze_chunk(chunk)
                prog.update(task, advance=1)

        self.index.determine_doc_type()
        self.index.deduplicate_sections()

        # Show how many problem numbers were mapped
        n_mapped = len(self.index.problem_number_map)
        console.print(
            f"[green]  Index complete — detected type: [bold]{self.index.doc_type.upper()}[/bold]"
            + (f"  |  {n_mapped} problem numbers mapped" if n_mapped else "")
            + "[/green]\n"
        )

    def _analyze_chunk(self, page_numbers: List[int]):
        page_map = "\n".join(
            f"  Image {i+1} = Page {p} of the document"
            for i, p in enumerate(page_numbers)
        )

        # FIX: request structured math_problems with explicit number field
        prompt = (
            f"You are analyzing a {self.total_pages}-page document.\n\n"
            f"PAGE MAPPING — use these EXACT page numbers in your JSON output:\n"
            f"{page_map}\n\n"
            f"CRITICAL: The 'page' field MUST be one of {page_numbers}. "
            f"Do NOT reset numbering to 1, 2, 3, 4 unless those are the actual values above.\n\n"
            "MATH PROBLEMS — include ONLY real equations, calculations, integrals, derivatives, "
            "algebraic expressions, geometry formulas, or statistical formulas.\n"
            "Do NOT list essay/conceptual questions or definitions as math problems.\n\n"
            "IMPORTANT: For each math problem, capture its PROBLEM NUMBER if one is printed "
            "(e.g. '108.', '23)', 'Problem 5'). This is critical for look-up accuracy.\n\n"
            "For each page identify:\n"
            "- topics: specific subjects\n"
            "- math_problems: list of objects with keys:\n"
            "    'number' (integer or null if no number visible),\n"
            "    'text'   (the equation/expression as plain text)\n"
            "- key_concepts: important terms, definitions, facts\n"
            "- visual_elements: any diagrams, tables, graphs, charts, maps, images\n"
            "- section: chapter/section title if visible, else null\n"
            "- content_type: 'math', 'text', or 'mixed'\n"
            "- has_images: true/false\n"
            "- has_tables: true/false\n"
            "- has_graphs: true/false\n"
            "- summary: 2-3 sentence description\n\n"
            f"Return ONLY valid JSON with exactly {len(page_numbers)} page entries "
            f"using page numbers {page_numbers}:\n"
            '{{"pages": [{{"page": PAGE_NUMBER, "topics": [], '
            '"math_problems": [{{"number": null, "text": ""}}], '
            '"key_concepts": [], "visual_elements": [], "section": null, '
            '"content_type": "text", "has_images": false, "has_tables": false, '
            '"has_graphs": false, "summary": ""}}]}}'
        )

        content = [{"type": "text", "text": prompt}]
        for p in page_numbers:
            if 1 <= p <= len(self.context_data):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.context_data[p-1]['image']}"}
                })

        try:
            resp = _api_call(self.client,
                model=MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise document analyzer. Return only valid JSON. "
                            f"Use EXACTLY the page numbers provided: {page_numbers}. "
                            "Do not renumber pages. "
                            "math_problems must be a list of objects: "
                            "[{\"number\": <int or null>, \"text\": \"<equation>\"}]"
                        )
                    },
                    {"role": "user", "content": content}
                ],
                temperature=0.05,
                max_tokens=3000
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
            if not raw.startswith('{'):
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    raw = json_match.group(0)
            data = json.loads(raw)

            returned_pages = data.get("pages", [])
            for i, page_data in enumerate(returned_pages):
                returned_num = page_data.get("page", 0)
                if returned_num not in page_numbers and i < len(page_numbers):
                    page_data["page"] = page_numbers[i]

                # Strip LaTeX from text fields
                if 'summary' in page_data and isinstance(page_data['summary'], str):
                    page_data['summary'] = _strip_latex(page_data['summary'])
                if 'section' in page_data and isinstance(page_data['section'], str):
                    page_data['section'] = _strip_latex(page_data['section'])
                # math_problems is now a list of dicts — strip LaTeX from 'text' field
                if 'math_problems' in page_data and isinstance(page_data['math_problems'], list):
                    cleaned = []
                    for prob in page_data['math_problems']:
                        if isinstance(prob, dict):
                            prob['text'] = _strip_latex(prob.get('text', ''))
                            cleaned.append(prob)
                        elif isinstance(prob, str):
                            # Tolerate old plain-string format from model
                            cleaned.append({'number': None, 'text': _strip_latex(prob)})
                    page_data['math_problems'] = cleaned

                actual_num = page_data.get("page", page_numbers[i] if i < len(page_numbers) else 0)
                self.index.add_page(actual_num, page_data)

        except Exception as e:
            console.print(f"[yellow]  Warning: chunk {page_numbers} — {e}[/yellow]")
            for p in page_numbers:
                self.index.add_page(p, {"page": p, "topics": [], "summary": ""})

    # ── QUERY HANDLING ────────────────────────────────────────────────────────

    def ask(self, user_input: str) -> str:
        ui  = user_input.strip()
        cmd = ui.lower().strip()

        if cmd in ('summarize', 'summary', 'summarize all', 'full summary'):
            return self._summarize_full()
        if m := re.match(r'summarize pages?\s*(\d+)\s*[-–to]\s*(\d+)', cmd):
            return self._summarize_range(int(m.group(1)), int(m.group(2)))
        if m := re.match(r'show pages?\s+(\d+)', cmd):
            return self._show_page(int(m.group(1)))
        if m := re.match(r'where is (.+)\??$', cmd):
            return self._find_topic(m.group(1).strip())

        if re.match(r'^summar\w*$', cmd) and not re.search(r'\d', cmd):
            return self._summarize_full()
        if m := re.match(r'(?:show\s+)?(?:pg|page)\s+(\d+)$', cmd):
            return self._show_page(int(m.group(1)))
        if m := re.match(r'(?:find|locate|search for|where)\s+(.+)$', cmd):
            return self._find_topic(m.group(1).strip())
        if m := re.match(r'(?:explain|describe|tell me about)\s+(.+)$', cmd):
            topic = m.group(1).strip()
            pages = self.index.find_pages_for_topic(topic)
            if pages:
                return self._query(ui)

        if re.search(r'\b(all|every|each)\b', cmd) and re.search(
            r'\b(images?|figures?|diagrams?|illustrations?|pictures?|photos?)\b', cmd
        ):
            return self._explain_all_images()
        if re.search(
            r'\b(explain|describe|list|show)\b.*\b(images?|figures?|diagrams?)\b', cmd
        ):
            return self._explain_all_images()

        return self._query(ui)

    def _query(self, ui: str) -> str:
        pages  = self._smart_page_selector(ui)
        q_type = self._classify_question(ui)

        if not self.messages:
            self.messages = [{"role": "system", "content": get_system_prompt(self.index.doc_type)}]

        text_history = [m for m in self.messages[1:]
                        if isinstance(m.get("content"), str)][-MAX_HIST * 2:]
        self.messages = [self.messages[0]] + text_history

        # FIX: for math questions, also include the next page in case the
        # problem or its context continues onto the next page
        if q_type == 'math' and pages:
            next_p = pages[-1] + 1
            if next_p <= self.total_pages and next_p not in pages:
                pages = (pages + [next_p])[:MAX_BATCH]

        context = self._build_context(ui, pages, q_type)
        self.messages.append({"role": "user", "content": context})

        is_math = q_type == 'math'
        try:
            resp = _api_call(self.client,
                model=MODEL_ID,
                messages=self.messages,
                temperature=0.05 if is_math else 0.25,
                max_tokens=4096
            )
        except RuntimeError as e:
            return f"**Rate limit reached.**\n\n{e}"
        except Exception as e:
            return f"**API error:** {e}"

        answer = _strip_latex(resp.choices[0].message.content)
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def _classify_question(self, text: str) -> str:
        ct = classify_content(text)
        if ct == 'text':
            pages = self.index.find_pages_for_topic(text)
            for p in pages:
                if self.index.pages.get(p, {}).get('content_type') == 'math':
                    return 'math'
        return ct

    def _smart_page_selector(self, query: str) -> List[int]:
        explicit = self._extract_explicit_pages(query)
        if explicit:
            return explicit

        # FIX: try to extract a problem number from the query and look it up
        # directly in the number map before falling back to text search.
        # Handles: "solve 108", "problem 108", "prob 108", "#108", "108"
        num_match = re.search(
            r'\b(?:problem|question|prob|q|#)?\.?\s*(\d+)\b',
            query, re.I
        )
        if num_match:
            num = int(num_match.group(1))

            # 1. Exact hit in the number map (fast path)
            if num in self.index.problem_number_map:
                p = self.index.problem_number_map[num]
                neighbors = []
                if p > 1:
                    neighbors.append(p - 1)
                neighbors.append(p)
                if p < self.total_pages:
                    neighbors.append(p + 1)
                return neighbors[:MAX_BATCH]

            # 2. Scan problem descriptions for the number
            for prob in self.index.math_problems:
                desc = prob.get('description', '')
                if re.search(rf'(?<!\d){num}(?!\d)', desc):
                    p = prob['page']
                    return list(dict.fromkeys(
                        [p] + ([p + 1] if p < self.total_pages else [p])
                    ))

        # 3. Topic / keyword match
        topic_pages = self.index.find_pages_for_topic(query)
        if topic_pages:
            return topic_pages

        # 4. Spread sample fallback
        if self.total_pages <= 4:
            return list(range(1, self.total_pages + 1))
        step = max(1, self.total_pages // 4)
        return [min(p, self.total_pages) for p in range(1, self.total_pages + 1, step)][:4]

    def _extract_explicit_pages(self, text: str) -> List[int]:
        pages = set()
        if m := re.search(r'pages?\s*(\d+)\s*[-–to]\s*(\d+)', text, re.I):
            s, e = int(m.group(1)), int(m.group(2))
            pages.update(p for p in range(s, min(e+1, self.total_pages+1)))
        elif m := re.search(r'pages?\s*(\d+(?:\s*,\s*\d+)+)', text, re.I):
            for n in m.group(1).split(','):
                p = int(n.strip())
                if 1 <= p <= self.total_pages:
                    pages.add(p)
        elif m := re.search(r'page\s+(\d+)\b', text, re.I):
            p = int(m.group(1))
            if 1 <= p <= self.total_pages:
                pages.add(p)
        return sorted(pages)[:MAX_BATCH]

    def _build_context(self, query: str, pages: List[int], q_type: str) -> list:
        meta = (
            f"DOCUMENT: '{self.doc_name}' — {self.total_pages} pages total\n"
            f"TYPE: {self.index.doc_type.upper()}\n"
            f"QUESTION TYPE DETECTED: {q_type.upper()}\n"
            f"PAGES VISIBLE TO YOU NOW: {pages}\n\n"
        )

        summaries = []
        for p in pages:
            pd = self.index.pages.get(p)
            if pd and pd.get('summary'):
                summaries.append(f"  Page {p}: {pd['summary']}")
        if summaries:
            meta += "PAGE CONTEXT:\n" + "\n".join(summaries) + "\n\n"

        if q_type == 'math':
            meta += (
                "INSTRUCTION: This is a MATH question. "
                "Look at the page images to find the exact problem. "
                "Solve step by step using Unicode math symbols only — NO LaTeX, NO dollar signs.\n\n"
            )
        else:
            meta += (
                "INSTRUCTION: Answer this question using information from the visible pages. "
                "Cite page numbers. If you see tables, graphs, or diagrams, describe what they show.\n\n"
            )

        meta += f"QUESTION: {query}"

        content = [{"type": "text", "text": meta}]
        for p in pages[:MAX_BATCH]:
            if 1 <= p <= len(self.context_data):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self.context_data[p-1]['image']}"}
                })
        return content

    # ── SPECIAL COMMANDS ──────────────────────────────────────────────────────

    def _summarize_full(self) -> str:
        if self.full_summary:
            return self.full_summary

        console.print("\n[dim cyan]  Synthesizing full document summary...[/dim cyan]")

        page_lines = []
        for p in sorted(self.index.pages.keys()):
            pd       = self.index.pages[p]
            s        = pd.get('summary', '')
            concepts = ', '.join(pd.get('key_concepts', [])[:5])
            if s:
                page_lines.append(
                    f"Page {p}: {s}" + (f" Key concepts: {concepts}" if concepts else "")
                )

        synthesis = (
            f"You are summarizing '{self.doc_name}' ({self.total_pages} pages).\n"
            f"Document type: {self.index.doc_type}\n\n"
            f"Page-by-page analysis:\n" + "\n".join(page_lines) + "\n\n"
            "Write a comprehensive summary organized by THEME (not page by page).\n"
            "Include:\n"
            "1. Overview (what this document is about)\n"
            "2. Major themes/topics covered\n"
            "3. Key concepts and definitions\n"
            "4. Notable data, findings, or results if any\n"
            "5. Structure/organization of the document\n\n"
            "Use markdown headers. Mention page numbers where relevant. "
            "Aim for thorough but clear — 400 to 700 words."
        )

        try:
            resp = _api_call(self.client,
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "You create clear, well-organized document summaries."},
                    {"role": "user", "content": synthesis}
                ],
                temperature=0.3,
                max_tokens=4096
            )
        except RuntimeError as e:
            return f"**Rate limit reached.**\n\n{e}"
        except Exception as e:
            return f"**API error:** {e}"

        self.full_summary = _strip_latex(resp.choices[0].message.content)
        return self.full_summary

    def _summarize_range(self, start: int, end: int) -> str:
        pages  = list(range(max(1, start), min(end + 1, self.total_pages + 1)))
        chunks = [pages[i:i+MAX_BATCH] for i in range(0, len(pages), MAX_BATCH)]
        parts  = []
        for chunk in chunks:
            result = self._query_with_pages(
                f"Summarize pages {chunk[0]} to {chunk[-1]} in detail.", chunk
            )
            parts.append(result)
        return "\n\n---\n\n".join(parts) if len(parts) > 1 else parts[0]

    def _explain_all_images(self) -> str:
        image_pages = sorted(self.index.has_images) if self.index.has_images \
                      else list(range(1, self.total_pages + 1))

        if not image_pages:
            return "No images or figures were detected in this document."

        console.print(f"\n[dim cyan]  Scanning {len(image_pages)} pages for images/figures...[/dim cyan]")

        chunks        = [image_pages[i:i+MAX_BATCH] for i in range(0, len(image_pages), MAX_BATCH)]
        batch_results = []

        for chunk in chunks:
            prompt = (
                f"Look at pages {chunk} carefully.\n"
                f"List and describe EVERY image, figure, diagram, photograph, map, chart, or illustration.\n"
                f"For each one:\n"
                f"- Figure number (e.g. Fig 6.1) if shown\n"
                f"- Page number\n"
                f"- Type (photograph, diagram, satellite image, map, etc.)\n"
                f"- What it shows and what concept it illustrates\n\n"
                f"Be thorough — do NOT skip any image. If a page has no images, skip it silently."
            )
            result = self._query_with_pages(prompt, chunk)
            if result:
                batch_results.append(result)

        if not batch_results:
            return "Could not retrieve image descriptions. Try asking about a specific page instead."
        if len(batch_results) == 1:
            return batch_results[0]

        combined         = "\n\n".join(batch_results)
        synthesis_prompt = (
            f"Below are descriptions of all figures and images from a document, collected page by page.\n"
            f"Reorganize into a single clean, numbered list of ALL figures in order.\n"
            f"Format each as:\n"
            f"**Figure X.X** (Page N) — [type]: [description]\n\n"
            f"Raw descriptions:\n{combined}"
        )
        try:
            resp = _api_call(self.client,
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "You organize and format figure descriptions clearly."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.2,
                max_tokens=4096
            )
            return _strip_latex(resp.choices[0].message.content)
        except Exception:
            return combined

    def _query_with_pages(self, prompt: str, pages: List[int]) -> str:
        q_type = 'text'
        if not self.messages:
            self.messages = [{"role": "system", "content": get_system_prompt(self.index.doc_type)}]

        context  = self._build_context(prompt, pages, q_type)
        messages = [self.messages[0], {"role": "user", "content": context}]

        try:
            resp = _api_call(self.client,
                model=MODEL_ID,
                messages=messages,
                temperature=0.2,
                max_tokens=2048
            )
            return _strip_latex(resp.choices[0].message.content)
        except RuntimeError as e:
            return f"**Rate limit reached.**\n\n{e}"
        except Exception as e:
            return f"**API error:** {e}"

    def _show_page(self, page_num: int) -> str:
        if not (1 <= page_num <= self.total_pages):
            return f"Page {page_num} does not exist. Document has {self.total_pages} pages."
        context = (
            f"Analyze page {page_num} in full detail.\n"
            f"Describe everything you can see: text content, any images or diagrams, "
            f"tables, math problems, headings, and key information.\n"
            f"Be thorough — this is a deep inspection of a single page."
        )
        return self._query(context)

    def _find_topic(self, topic: str) -> str:
        pages = self.index.find_pages_for_topic(topic)
        if not pages:
            return f"Could not find '{topic}' in the document index. Try asking a direct question about it."
        result = [f"**'{topic}'** found on: Pages {', '.join(str(p) for p in pages)}\n"]
        for p in pages:
            pd = self.index.pages.get(p, {})
            s  = pd.get('summary', '')
            if s:
                result.append(f"**Page {p}:** {s}")
        return "\n".join(result)

    # ── INDEX DISPLAY ──────────────────────────────────────────────────────────

    def render_index(self):
        md = self._build_index_markdown()
        console.print(
            Panel(
                Markdown(md),
                title="Document Index (All Pages Analyzed)",
                border_style="blue"
            )
        )
        console.print("\n[bold]Commands:[/bold]")
        console.print("  - [cyan]summarize[/cyan] - Complete document summary")
        console.print("  - [cyan]summarize pages X-Y[/cyan] - Summarize a page range")
        console.print("  - [cyan]explain [topic][/cyan] - Detailed topic explanation")
        console.print("  - [cyan]where is [topic]?[/cyan] - Find topic location")
        console.print("  - [cyan]show page X[/cyan] - View specific page")
        if self.index.math_problems:
            console.print("  - [cyan]solve problem X[/cyan] - Solve a math problem")
        console.print("  - [cyan]exit[/cyan] - Quit\n")

    def _build_index_markdown(self) -> str:
        md  = "## Document Index\n\n"
        md += f"**Document:** {self.doc_name}\n"
        md += f"**Total Pages:** {self.total_pages} (ALL pages have been analyzed)\n"
        md += f"**Type:** {self.index.doc_type.upper()}\n"
        if self.index.problem_number_map:
            nums = sorted(self.index.problem_number_map.keys())
            md += f"**Problem Numbers Indexed:** {nums[0]}–{nums[-1]} ({len(nums)} problems)\n"
        md += "\n"

        if self.index.sections:
            seen_pages: set = set()
            unique_sections = []
            for s in sorted(self.index.sections, key=lambda x: x['page']):
                if s['page'] not in seen_pages:
                    seen_pages.add(s['page'])
                    unique_sections.append(s)
            md += f"### Document Structure ({len(unique_sections)} sections)\n\n"
            for s in unique_sections[:15]:
                md += f"- Page {s['page']}: {s['title']}\n"
            if len(unique_sections) > 15:
                md += f"- ... and {len(unique_sections)-15} more sections\n"
            md += "\n"

        if self.index.topics:
            topic_items = sorted(self.index.topics.items(), key=lambda x: (-len(x[1]), x[0]))
            md += f"### Main Topics ({len(self.index.topics)} found)\n\n"
            for topic, pages in topic_items:
                pstr = ", ".join(str(p) for p in sorted(pages))
                md += f"- {topic} *(p.{pstr})*\n"
            md += "\n"

        if self.index.math_problems:
            md += f"### Math Problems ({len(self.index.math_problems)} found)\n\n"
            for i, prob in enumerate(self.index.math_problems[:12], 1):
                num_str = f"#{prob['number']} " if prob.get('number') else ""
                desc    = prob['description'][:80] + ("..." if len(prob['description']) > 80 else "")
                md += f"{i}. *(p.{prob['page']})* {num_str}{desc}\n"
            if len(self.index.math_problems) > 12:
                md += f"... and {len(self.index.math_problems)-12} more\n"
            md += "\n"

        vis_parts = []
        if self.index.has_images:
            vis_parts.append(f"Images on p.{','.join(str(p) for p in sorted(self.index.has_images)[:5])}")
        if self.index.has_tables:
            vis_parts.append(f"Tables on p.{','.join(str(p) for p in sorted(self.index.has_tables)[:5])}")
        if self.index.has_graphs:
            vis_parts.append(f"Graphs on p.{','.join(str(p) for p in sorted(self.index.has_graphs)[:5])}")
        if vis_parts:
            md += "### Visual Elements\n\n"
            for v in vis_parts:
                md += f"- {v}\n"
            md += "\n"

        md += "### Available Commands\n\n"
        md += "- **summarize** - Get comprehensive summary of entire document\n"
        md += "- **summarize pages X-Y** - Summarize specific page range\n"
        md += "- **explain [topic]** - Detailed explanation of any topic\n"
        md += "- **where is [topic]?** - Find which pages discuss a topic\n"
        md += "- **show page X** - View and analyze specific page\n"
        if self.index.math_problems:
            md += "- **solve problem X** - Solve math problems\n"

        return md


# ─────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────
def _print_rule(title: str):
    console.print(Rule(f"[bold cyan] {title} [/bold cyan]", style="dim blue"))

def _print_response(content: str, question: str):
    console.print(
        Panel(
            Markdown(content),
            title="Response",
            border_style="blue"
        )
    )

def _print_header():
    console.print(
        Panel.fit(
            "[bold cyan]Document Intelligence Assistant[/bold cyan]\n"
            "[dim]Full Document Analysis & Summarization[/dim]",
            border_style="cyan"
        )
    )

def _print_thinking():
    return console.status("[cyan]Thinking...[/cyan]", spinner="dots")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    _print_header()

    assistant = DocumentAssistant()

    file_path = console.input("[bold yellow]  Enter file path (PDF / PNG / JPG): [/bold yellow]").strip()
    console.print()

    if not assistant.load_file(file_path):
        sys.exit(1)

    assistant.build_index()
    assistant.render_index()

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ('exit', 'quit', 'q', 'bye'):
            console.print("[yellow]Session ended.[/yellow]")
            break

        with _print_thinking():
            response = assistant.ask(user_input)

        _print_response(response, user_input)


if __name__ == "__main__":
    main()