import json
import re

_GUTENBERG_END = re.compile(r'\*{3}\s*END\b.*', re.DOTALL)

def strip_gutenberg_boilerplate(text, stop_markers=None):
    text = _GUTENBERG_END.split(text)[0]
    if stop_markers:
        for marker in stop_markers:
            text = text.split(marker)[0]
    return text


class BaseExtractor:
    """
    Base class for extractors. Subclasses should implement the extract_pairs method to return a list of
    {"q": ..., "a": ...} dicts.
    """
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def extract_pairs(self, text):
        raise NotImplementedError

    def run(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            text = f.read()

        pairs = self.extract_pairs(text)

        records = [
            {
                "conversations": [
                    {"role": "user", "content": pair["q"]},
                    {"role": "assistant", "content": pair["a"]}
                ]
            }
            for pair in pairs
        ]

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"Extracted {len(pairs)} Q&A pairs → {self.output_path}")


class CatechismExtractor(BaseExtractor):
    """
    Many catechisms are formatted with 'Question. Question text' followed by 'Answer. Answer text',
    repeated for each Q&A pair. This extractor handles that format.
    """
    PATTERN = re.compile(
        r'(?:Question|Q)\.\s+(.+?)\n\n(?:Answer|A)\.\s+(.+?)(?=\n\n(?:Question|Q)\.|\Z)',
        re.MULTILINE | re.DOTALL
    )

    def preprocess(self, text):
        text = re.sub(r'^A,\s', 'A. ', text, flags=re.MULTILINE)
        return text

    def extract_pairs(self, text):
        text = self.preprocess(text)

        pairs = []
        for q_text, a_text in self.PATTERN.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class BrewersGuideExtractor(CatechismExtractor):
    pass


class AstronomyExtractor(CatechismExtractor):
    """
    Astronomy catechism. Questions are numbered ('Q. 1. text') and some Q&As are on a single line.
    '* A.' is an OCR variant of 'A.'. Short section/page headers are interspersed.
    """
    def preprocess(self, text):
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)
        text = re.sub(r'^Q\. \d+\. ', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'\* A\.', 'A.', text)
        text = re.sub(r'^A,\s', 'A. ', text, flags=re.MULTILINE)  # A, OCR for A.
        # Inline Q. and A. splits for multi-pair paragraphs
        text = re.sub(r'([.,?])\s*\S*\s+(Q\. )', r'\1\n\n\2', text)
        text = re.sub(r'([.?])\s+(A[.,] )(?=[A-Z][a-z])', r'\1\n\nA. ', text)
        return text


class ChemistryExtractor(CatechismExtractor):
    """
    Chemistry catechism. Q. OCRs as Q.. / Q,, / Q, / Gt. / Gl. throughout. Contains OCR hyphens
    and page headers. Many Q/A pairs are crammed onto a single line with no blank-line separator;
    these are split into proper paragraphs before parsing.
    """
    def preprocess(self, text):
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        # Normalize all OCR variants of 'Q.' to 'Q. '
        text = re.sub(r'^Q[.,]+\s', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'^G[tl]\.\s', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'^d\.\s', 'Q. ', text, flags=re.MULTILINE)  # d. OCR for Q.
        text = re.sub(r"^'\s*Q\.\s", 'Q. ', text, flags=re.MULTILINE)  # ' Q. (leading quote)
        # Normalize 'A,' (comma OCR for period) at line-start
        text = re.sub(r'^A,\s', 'A. ', text, flags=re.MULTILINE)
        # Strip inline noise tokens before A. markers (e.g. 'IP A.' page artifacts)
        text = re.sub(r'\b[A-Z]{1,3} (A[.,] )', r'?\n\nA. ', text)
        # OCR '1' and '^' appear as '?' at end of questions before inline A.
        text = re.sub(r'(\w) 1 (A[.,] )', r'\1?\n\nA. ', text)
        text = re.sub(r' \^ (A[.,] )', r'?\n\nA. ', text)
        # Bullet or single-quote before inline Q. (with or without space)
        text = re.sub(r'[•\']\s*(Q\. )', r'?\n\n\1', text)
        # Split inline Q./A. transitions: insert blank line before Q. or A. that
        # immediately follows a sentence-ending punctuation on the same line
        text = re.sub(r'([?.]) (Q\. )', r'\1\n\n\2', text)
        text = re.sub(r'([.,?]) (Q\. )', r'\1\n\n\2', text)
        text = re.sub(r'([.?]) d\. ', r'\1\n\nQ. ', text)   # inline d. after sentence
        text = re.sub(r'([?.])\s+(A[.,] )', r'\1\n\nA. ', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        return '\n\n'.join(paragraphs)


class NewYorkBarExtractor(CatechismExtractor):
    """
    NY Bar Exam study guide. Standard Q./A. catechism with OCR hyphens and topic headers.
    Q. OCRs with leading noise chars: 'I Q.', '•Q.', 'A^Q.', '\\ Q.', '; Q.', 'J Q.', '"^ Q.'
    A. OCRs as 'A,' (comma) inline after questions.
    """
    def preprocess(self, text):
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        # Normalize specific Q. OCR variants at line start (noise chars prepended by OCR)
        text = re.sub(r'^I\s+Q\.\s', 'Q. ', text, flags=re.MULTILINE)    # I Q.
        text = re.sub(r'^J\s+Q\.\s', 'Q. ', text, flags=re.MULTILINE)    # J Q.
        text = re.sub(r'^[•]\s*Q\.\s', 'Q. ', text, flags=re.MULTILINE)  # •Q.
        text = re.sub(r'^[\\]\s+Q\.\s', 'Q. ', text, flags=re.MULTILINE) # \ Q.
        text = re.sub(r'^["\'^]+\s*Q\.\s', 'Q. ', text, flags=re.MULTILINE)  # "^ Q. ^Q.
        text = re.sub(r'^[A-Z]\^Q\.\s', 'Q. ', text, flags=re.MULTILINE) # A^Q.
        # Normalize 'A,' inline answer marker after question mark
        text = re.sub(r'\?\s+A,\s', '?\n\nA. ', text)
        # Split inline Q. transitions after citation/answer endings
        for _ in range(3):
            prev = text
            text = re.sub(r'([.,?!;])\s*["\'^•\\I]*\s*(Q\. )', r'\1\n\n\2', text)
            if text == prev:
                break
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        return '\n\n'.join(paragraphs)


class SchoolBulletinExtractor(BaseExtractor):
    """
    Dime Question Book. Numbered questions with 'Ans.' or 'Ans—' answer marker.
    Multi-paragraph answers; OCR hyphens and inline page headers.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)

        pattern = re.compile(
            r'^\s*\d+\.\s+(.+?)\n\nAns[.\-—]+\s*(.+?)(?=^\s*\d+\.|\Z)',
            re.MULTILINE | re.DOTALL,
        )

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r'\s+', ' ', q_text.strip())
            answer = re.sub(r'\s+', ' ', a_text.strip().lstrip('—–-'))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class GrammarExtractor(CatechismExtractor):
    """
    Grammar catechism. 4457-line preamble before Q&A section. Heavily OCR-corrupted:
    Q-marker variants include '^;', '^:', '§1;', '§1:', 'SI;', 'SI:', 'Question:',
    '% ', '|>;', ':^;', 'i^;', 'S>;' and others. A-marker variants include 'A:', 'J:',
    '/I:', '//;', './;', 'Answer:'. Also, '?' OCRs as '.^^', '.f^', '.f*', '}.
    Markers appear at line-start and inline. Strategy: normalize all known variants,
    split inline pairs on '?' + A-marker, then handle remaining noise in final cleanup.
    """

    def preprocess(self, text):
        qa_start = re.search(r'QUESTIONS AND ANSWERS', text)
        if qa_start:
            text = text[qa_start.start():]
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)

        # --- Normalize OCR '?' variants ---
        text = re.sub(r'\s*\.\^\^', '?', text)       # .^^ → ?
        text = re.sub(r'\s*\.[f]\^', '?', text)       # .f^ → ?
        text = re.sub(r'\s*\.[f]\*', '?', text)       # .f* → ?
        text = re.sub(r'\s*\.\^(?= )', '?', text)     # .^ followed by space → ?
        text = re.sub(r"\.['\"]?\^(?= )", '?', text)  # .'^ or .^ → ?
        text = re.sub(r' \}', '?', text)              # } → ?

        # --- Normalize Q-marker variants at line-start to 'Q. ' ---
        # caret-glyph variants: ^; ^: ^. ^* ^- ^j etc.
        text = re.sub(r'^\^[^A-Za-z0-9\s]+\s*', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'^\^[jl]\s+', 'Q. ', text, flags=re.MULTILINE)  # ^j ^l
        # parenthesized caret: (^.
        text = re.sub(r'^\([^A-Za-z0-9\s]+\s*', 'Q. ', text, flags=re.MULTILINE)
        # percent sign
        text = re.sub(r'^[.]*% +', 'Q. ', text, flags=re.MULTILINE)
        # section-number variants: §1; §1: SI; SI:
        text = re.sub(r'^[§S][I1][;:]\s*', 'Q. ', text, flags=re.MULTILINE)
        # italic-i + caret: i^;
        text = re.sub(r'^i\^[;:]\s*', 'Q. ', text, flags=re.MULTILINE)
        # pipe variants: |>; :^;  S>;
        text = re.sub(r'^\|>[;:]\s*', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'^:[^A-Za-z0-9\s]+\s*', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'^S>[;:]\s*', 'Q. ', text, flags=re.MULTILINE)
        # Question: or 'Question: (stray leading quote)
        text = re.sub(r"^['\"]?Question:\s*", 'Q. ', text, flags=re.MULTILINE)
        # ?l: (OCR of ^;) at line-start
        text = re.sub(r'^\?l:\s*', 'Q. ', text, flags=re.MULTILINE)

        # --- Normalize A-marker variants at line-start to 'A. ' ---
        text = re.sub(r'^A:\s*', 'A. ', text, flags=re.MULTILINE)
        text = re.sub(r'^J:\s*', 'A. ', text, flags=re.MULTILINE)
        text = re.sub(r'^/[I/][;:]\s*', 'A. ', text, flags=re.MULTILINE)  # /I: //;
        text = re.sub(r'^\./[;:]\s*', 'A. ', text, flags=re.MULTILINE)    # ./;
        text = re.sub(r'^Answer:\s*', 'A. ', text, flags=re.MULTILINE)

        # --- Split inline Q/A pairs ---
        # After normalization, inline pairs look like '? A. ' or '? A: ' etc.
        # Some inline A-markers are not at line-start so weren't normalised above;
        # handle them specifically after '?'.
        #  ^: after ? = A-marker (^: at line-start = Q-marker, already handled above)
        text = re.sub(r'\? \^:\s*', '?\n\nA. ', text)
        #  \i: after ? = A-marker
        text = re.sub(r'\? \\i:\s*', '?\n\nA. ', text)
        #  ?l: after ? = A-marker  (?l: is OCR of A:)
        text = re.sub(r'\? \?l:\s*', '?\n\nA. ', text)
        #  ^j after ? = A-marker
        text = re.sub(r'\? \^j\s*', '?\n\nA. ', text)
        #  y^. after ? = A-marker (OCR of A.)
        text = re.sub(r'\? y\^[.]\s*', '?\n\nA. ', text)
        # Strip any other stray non-alpha chars between ? and A-marker
        text = re.sub(r'\? [^A-Za-z0-9\n]+ A[:.] ', r'?\n\nA. ', text)
        # A, (comma OCR of period) as A-marker after ?
        text = re.sub(r'\? A, ', r'?\n\nA. ', text)
        # Also treat lone ^ before A-marker as ? (e.g. 'employed ^ A:')
        text = re.sub(r' \^ A[:.] ', r'?\n\nA. ', text)
        # Handle y^. as inline A-marker following ? or :
        text = re.sub(r'[?:] y\^[.] ', r'?\n\nA. ', text)
        # Inline ^: after sentence-end = Q-marker; inline ^; after . = Q-marker
        text = re.sub(r'([.,]) \^[;:] ', r'\1\n\nQ. ', text)
        # Standard inline splits, repeated until stable
        for _ in range(10):
            prev = text
            text = re.sub(r'([.?:,]) Q\. ', r'\1\n\nQ. ', text)
            text = re.sub(r'([.?:,]) A\. ', r'\1\n\nA. ', text)
            text = re.sub(r'([.?:,]) A: ', r'\1\n\nA. ', text)
            text = re.sub(r'([.?:,]) J: ', r'\1\n\nA. ', text)
            text = re.sub(r'([.?:,]) Answer: ', r'\1\n\nA. ', text)
            # Also handle inline % Q-marker and inline ^; ^: Q-markers
            text = re.sub(r'([.?:,]) % ', r'\1\n\nQ. ', text)
            text = re.sub(r'([.?:,]) \^[;:] ', r'\1\n\nQ. ', text)
            if text == prev:
                break

        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        return '\n\n'.join(paragraphs)



class LogicExtractor(CatechismExtractor):
    """
    Logic catechism. Severe extra-space OCR ('Q.  What  is  the  difference').
    'Qaesllon.' OCRs as 'Question.'; 'J.' OCRs as 'A.' in places.
    Many OCR variants for 'A.': A, Ai. yl. yi. //. ^'. ui. and bullet prefix.
    Some Q/A pairs have no blank line between Q and A lines.
    """
    def preprocess(self, text):
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'^Qaesllon\.', 'Q.', text, flags=re.MULTILINE)
        text = re.sub(r'^Q,', 'Q.', text, flags=re.MULTILINE)   # Q, OCR for Q.
        text = re.sub(r'^Q\s+\.', 'Q.', text, flags=re.MULTILINE)  # Q .What OCR for Q.
        text = re.sub(r'^O\.', 'Q.', text, flags=re.MULTILINE)   # O. OCR for Q.
        text = re.sub(r'^0\.', 'Q.', text, flags=re.MULTILINE)   # 0. (digit) OCR for Q.
        text = re.sub(r'^\(>\.',  'Q.', text, flags=re.MULTILINE)  # (>. OCR for Q.
        text = re.sub(r'^J\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^A,', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^Ai\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^yl\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^yi\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^//\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^\^\'\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^ui\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^[•*]\s+A\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'^\^\.', 'A.', text, flags=re.MULTILINE)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        # Split inline Q./A. pairs — Q, OCR remains inline after paragraph merging
        text = re.sub(r'([.,?;])\s*[\'"]?\s*Q[,.]\s+', r'\1\n\nQ. ', text)
        # Only split on inline A. when followed by a word (capital + lowercase), not
        # single-letter abbreviations like 'A. E. O.' syllogism notation.
        text = re.sub(r'([.,?;])\s+(A\. )(?=[A-Z][a-z])', r'\1\n\n\2', text)
        text = re.sub(r'(^Q\. .+)\n(A\. )', r'\1\n\n\2', text, flags=re.MULTILINE)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        return '\n\n'.join(paragraphs)


class PatriotismExtractor(CatechismExtractor):
    """
    Patriotism catechism. Standard Q./A. format; '-4.' OCRs as 'A.' in places.
    OCR hyphens and chapter headers interspersed.
    """
    def preprocess(self, text):
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        text = re.sub(r'^-4\.', 'A.', text, flags=re.MULTILINE)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        return '\n\n'.join(paragraphs)


class SeeleysExtractor(BaseExtractor):
    """
    Seeley's Question Book. Numbered questions with positional answers (no 'A.' marker).
    Q and A may be on the same line or split by a blank line; split at first '?'.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)

        blocks = re.split(r'(?m)^\s*\d+\.\s+', text)

        pairs = []
        for block in blocks:
            if '?' not in block:
                continue
            idx = block.index('?')
            question = re.sub(r'\s+', ' ', block[:idx + 1].strip())
            answer = re.sub(r'\s+', ' ', block[idx + 1:].strip())
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class BotanyExtractor(CatechismExtractor):
    """
    Botany catechism. Numbered questions ('Q. 1. text'); '@.' OCRs as 'Q.' in places.
    Heavy OCR-garbage preamble (~450 lines) before Q&A content begins.
    """
    def preprocess(self, text):
        first_q = re.search(r'(?m)^Q\. \d+\.', text)
        if first_q:
            text = text[first_q.start():]
        text = re.sub(r'^@\.', 'Q.', text, flags=re.MULTILINE)
        text = re.sub(r'^_\s*([QA])[.,]', r'\1.', text, flags=re.MULTILINE)  # _A. / _ A, / _ Q. italic OCR
        text = re.sub(r'^Q\. \d+\. ', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)
        # Ensure A. always opens its own paragraph regardless of what preceded it
        text = re.sub(r'([^\n])\n(A\. )', r'\1\n\n\2', text)
        # Also ensure Q. opens its own paragraph (handles headers/page-numbers before Q.)
        text = re.sub(r'([^\n])\n(Q\. )', r'\1\n\n\2', text)
        # Inline Q. after sentence-ending punct
        text = re.sub(r'([.,?!;])\s*[-—~\d]*\s*(Q\. )', r'\1\n\n\2', text)
        return text


class EthicsExtractor(CatechismExtractor):
    """
    Ethics catechism. Questions prefixed 'N. Q.' (number then Q.); answers prefixed 'A.'.
    OCR hyphens and page headers interspersed.
    """
    def preprocess(self, text):
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        # Normalize Q. with leading non-alpha chars at line start (.28. Q., ~213. Q.)
        text = re.sub(r'^[^A-Za-z0-9]*\d+\.\s+Q\.', 'Q.', text, flags=re.MULTILINE)
        text = re.sub(r'^A[,;]', 'A.', text, flags=re.MULTILINE)
        # Ensure Q./A. always open their own paragraphs so section headers get isolated
        text = re.sub(r'([^\n])\n(Q\. )', r'\1\n\n\2', text)
        text = re.sub(r'([^\n])\n(A\. )', r'\1\n\n\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)
        # Inline splits: numbered Q. (e.g. '— 288. Q.') → strip number, insert blank line
        text = re.sub(r'[^A-Za-z0-9\n]\s*\d+\.\s+Q\.\s', '\n\nQ. ', text)
        text = re.sub(r'([.,?!;])\s*[-—~]*\s*(Q\. )', r'\1\n\n\2', text)
        text = re.sub(r'([.?])\s+(A\. )(?=[A-Z][a-z])', r'\1\n\n\2', text)
        return text


class ElectricityExtractor(BaseExtractor):
    """
    Electrical catechism. Numbered questions ('1. What is X?') with answers in the next paragraph(s).
    Unlike Agriculture/CommonCore, Q and A are in separate paragraphs rather than a single block.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)

        blocks = re.split(r'(?m)^\s*\d+\.\s+', text)

        pairs = []
        for block in blocks:
            if '?' not in block:
                continue
            idx = block.index('?')
            question = re.sub(r'\s+', ' ', block[:idx + 1].strip())
            answer = re.sub(r'\s+', ' ', block[idx + 1:].strip())
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class MythologyExtractor(BaseExtractor):
    """
    Mythology catechism. Questions marked with '(1)', '(2)', etc.
    Some entries have a blank line before the answer; others don't.
    Split at first '?' to reliably separate question from answer.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)

        blocks = re.split(r'(?m)^\(\d+\)\s+', text)
        pairs = []
        for block in blocks:
            if '?' not in block:
                continue
            idx = block.index('?')
            question = re.sub(r'\s+', ' ', block[:idx + 1].strip())
            answer = re.sub(r'\s+', ' ', block[idx + 1:].strip())
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class ConstitutionExtractor(CatechismExtractor):
    """
    Constitution catechism. Standard Q./A. format with many OCR variants.
    Q. OCRs as '<£,', '<£.', '<£ ', '<j>.', '(j>.', '(£.', '(^.', '<^.', 'q.', 'Q ' (missing
    period), 'Q,'.
    A. OCRs as 'A,', 'A*', 'A .', 'Jl.', 'JL.', 'ei.', '*#.', '«/2.', '«/Z.' (garbled).
    Some Q/A pairs are concatenated on a single line with no blank line between them.
    Inline page numbers and hyphenation artifacts present.
    """
    def preprocess(self, text):
        # Repair hyphenation before any other normalization
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        # Normalize OCR variants of 'Q.' at line starts
        # <£, / <£. / <£<space>  and  <^.
        text = re.sub(r'^<[£\^][,. ]', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'^<[£\^]\.',    'Q.',  text, flags=re.MULTILINE)
        # <j>. (j with angle brackets)
        text = re.sub(r'^<j>\.',       'Q.',  text, flags=re.MULTILINE)
        # (j>. / (j). / (£. / (^.
        text = re.sub(r'^\([j£\^][>)]?\.', 'Q.', text, flags=re.MULTILINE)
        text = re.sub(r'^q\.', 'Q.', text, flags=re.MULTILINE)
        text = re.sub(r'^Q ', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'^Q,', 'Q.', text, flags=re.MULTILINE)
        # Normalize OCR variants of 'A.' at line starts
        text = re.sub(r'^A[,*]', 'A.', text, flags=re.MULTILINE)    # A, and A*
        text = re.sub(r'^A \.',  'A.', text, flags=re.MULTILINE)     # A . (space before period)
        text = re.sub(r'^J[lL]\.', 'A.', text, flags=re.MULTILINE)  # Jl. and JL.
        text = re.sub(r'^ei\.', 'A.', text, flags=re.MULTILINE)      # ei. (OCR garble)
        text = re.sub(r'^\*#\.', 'A.', text, flags=re.MULTILINE)     # *#. (OCR garble)
        text = re.sub(r'^«/[2Z]\.', 'A.', text, flags=re.MULTILINE) # «/2. and «/Z. (OCR garble)
        # Strip standalone page numbers and trailing page-number artifacts on Q lines
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(^Q\.[^\n]+\?) \d+\*?\s*$', r'\1', text, flags=re.MULTILINE)
        # Split concatenated Q/A pairs on the same line
        # Handle 'A. (leading tick from OCR) inline before the standard split
        text = re.sub(r"([.?])\s+'A\.\s+([A-Z])", r'\1\n\nA. \2', text)
        text = re.sub(r'([.?])\s+(Q\.)\s+', r'\1\n\n\2 ', text)
        text = re.sub(r'([.?])\s+(A\.)\s+([A-Z])', r'\1\n\nA. \3', text)
        # Insert blank line between Q line and A line when missing
        text = re.sub(r'(^Q\. .+)\n(A\. )', r'\1\n\n\2', text, flags=re.MULTILINE)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return '\n\n'.join(paragraphs)


class CivilWarExtractor(BaseExtractor):
    """
    Civil War Q&A. Questions marked 'Q. —', answers marked 'A. —' (with em-dash).
    Multi-paragraph answers, OCR hyphens, and 'Questions and Answers' page headers.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        text = re.sub(r'^Questions and Answers\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'\n{3,}', '\n\n', text)

        pattern = re.compile(r'Q\. [—\-]\s+(.+?)\n\nA\. [—\-]\s+(.+?)(?=\n\nQ\. [—\-]|\Z)', re.DOTALL)

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r'\s+', ' ', q_text.strip())
            answer = re.sub(r'\s+', ' ', a_text.strip())
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class EngineeringExtractor(CatechismExtractor):
    """
    Engineering catechism. Standard Q./A. format with OCR hyphens and running page headers.
    Running headers take the form "N roper's catechism for" / "STEAM ENGINEERS AND ELECTRICIANS. N"
    and appear throughout — including mid-answer — causing Q/A bleed if not removed before
    paragraph splitting.
    """
    # Matches the two alternating running-header forms used throughout this book:
    #   "N roper's catechism for" / "N ROPER'S CATECHISM FOR" / "O ROPER S CATECHISM FOR"
    #   "STEAM ENGINEERS AND ELECTRICIANS. N"
    # Case-insensitive to catch OCR variants like "'A roper's catechism for".
    _RUNNING_HEADER = re.compile(
        r'^(?:'
        r'[^\n]*\broper\b[^\n]*\bcatechism\b[^\n]*'
        r'|STEAM ENGINEERS AND ELECTRICIANS\.?[^\n]*'
        r')$',
        re.MULTILINE | re.IGNORECASE,
    )

    def preprocess(self, text):
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        # Strip running page headers before paragraph splitting so that answers
        # split across a header are re-joined into a single paragraph.
        text = self._RUNNING_HEADER.sub('', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        paragraphs = re.split(r'\n\n+', text)
        # Merge continuation paragraphs back into the preceding paragraph.
        # After header removal, any paragraph that neither starts with "Q."/"A."
        # nor looks like a section title is the tail of an answer split by a
        # page header; merge it back into the preceding block.
        merged = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if _is_symbological_header(p):
                continue
            if merged and not re.match(r'^[QA]\.', p):
                merged[-1] = merged[-1] + ' ' + p
            else:
                merged.append(p)
        text = '\n\n'.join(merged)
        # Split paragraphs where Q. and A. are inline with no blank line between them.
        # Also handle inline noise tokens before Q. (page numbers, bullets, stray chars).
        for _ in range(5):
            prev = text
            # Q paragraph containing inline A. → split at A.
            text = re.sub(r'(^Q\.[^\n]+\S) (A[.,] )(?=[A-Z])', r'\1\n\nA. ', text, flags=re.MULTILINE)
            # A paragraph (or any) containing inline Q. → split before Q.
            text = re.sub(r'([.,?!;])\s*[-—~\'"`^*■•!l\s]*\s*(Q\. )', r'\1\n\n\2', text)
            # Page refs like 'D*' before Q. (geometry/mechanics notation)
            text = re.sub(r'\s+[A-Z]\w*\*\s+(Q\. )', r'\n\n\1', text)
            # Comma directly attached to Q. (no space): ,Q. → ,\n\nQ.
            text = re.sub(r',Q\. ', r',\n\nQ. ', text)
            if text == prev:
                break
        return text


class MusicExtractor(BaseExtractor):
    """
    Music catechism. Questions are plain lines ending with '?', answers follow after a blank line.
    No Q./A. markers. Page headers and section titles are interspersed.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)

        pattern = re.compile(r'^([^\n]+\?\.?)\n\n(.+?)(?=\n\n[^\n]+\?|\Z)', re.MULTILINE | re.DOTALL)

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip())
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class AgricultureExtractor(BaseExtractor):
    """
    Agriculture catechism. Numbered questions ('1. What is X?') followed by answer paragraphs.
    Section headers and OCR hyphens are interspersed.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)

        pattern = re.compile(r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)", re.MULTILINE | re.DOTALL)

        pairs = []
        for _num, block in pattern.findall(text):
            idx = block.rfind("?")
            if idx == -1:
                continue
            question = re.sub(r" {2,}", " ", block[:idx + 1].strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", block[idx + 1:].strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


def _is_symbological_header(paragraph):
    """
    Detect if a paragraph is a header like 'Question 1.' or 'Q.1' that should be removed before parsing.
    """
    lines = [l.strip() for l in paragraph.strip().split('\n') if l.strip()]
    if not lines:
        return True
    if re.match(r'^(Question|Answer|Q|A)\.', lines[0]):
        return False
    if len(lines) > 3:
        return False
    text = ' '.join(lines)
    alpha = sum(1 for c in text if c.isalpha())
    upper = sum(1 for c in text if c.isupper())
    return alpha > 0 and upper / alpha > 0.65 and len(text) < 80


class SymbologicalExtractor(CatechismExtractor):
    """
    The Symbological is formatted with section headers like 'Question 1.' that break up the text. 
    We need to remove those before parsing.
    """
    def preprocess(self, text):
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p for p in paragraphs if not _is_symbological_header(p)]
        text = '\n\n'.join(paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'([.,?!;])\s*[-—~\d]*\s*(Q\. )', r'\1\n\n\2', text)
        return text

class CommonCoreExtractor(BaseExtractor):
    """
    Common Core is formatted with numbered questions like '1. Question text' followed by answer text,
    repeated for each Q&A pair.
    """
    def extract_pairs(self, text):
        pattern = re.compile(r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)", re.MULTILINE | re.DOTALL)
        matches = pattern.findall(text)

        pairs = []
        for _num, block in matches:
            idx = block.rfind("?")
            if idx == -1:
                continue
            question = re.sub(r" {2,}", " ", block[:idx + 1].strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", block[idx + 1:].strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs

class FamiliarThingsExtractor(BaseExtractor):
    """
    Familiar Things is formatted with questions as single lines ending with '?' followed by a blank line and
    then the answer text, repeated for each Q&A pair. Answers may include indented glossary definitions that
    should be included in the answer text.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text, stop_markers=['\nINDEX.'])
        pattern = re.compile(r'^([^\n]+\?)\n\n(.+?)(?=\n\n[^\n]+\?|\Z)', re.MULTILINE | re.DOTALL)

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = q_text.strip()
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs

class Questions1001Extractor(BaseExtractor):
    """
    1001 Questions is formatted with numbered questions like '1. Question text' followed by answer text starting
    with 'Answer.' on the next line, repeated for each Q&A pair.
    """
    def extract_pairs(self, text):
        pattern = re.compile(r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)", re.MULTILINE | re.DOTALL)

        pairs = []
        for _num, block in pattern.findall(text):
            idx = block.rfind("?")
            if idx == -1:
                continue
            question = re.sub(r" {2,}", " ", block[:idx + 1].strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", block[idx + 1:].strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs
    
class StokersExtractor(BaseExtractor):
    """
    Stokers is formatted with numbered questions like '1. Question text' followed by answer text starting
    with 'Answer.' on the next line, repeated for each Q&A pair. However, the question text may include inline
    page markers like [8] that should be removed before parsing.
    """
    def extract_pairs(self, text):
        # Strip inline page markers like [8] before parsing
        text = re.sub(r'\[\d+\]', '', text)

        pattern = re.compile(
            r'^\d+\.\s+Question\.—(.+?)\n\nAnswer\.—(.+?)(?=\n\n\d+\.\s+Question\.|\Z)',
            re.MULTILINE | re.DOTALL
        )

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class LaborersExtractor(BaseExtractor):
    """
    The Laborer's Catechism. Format: each paragraph is a complete Q&A block: 'Question? Answer.'
    Page headers and lesson titles are interspersed as noise. OCR hyphens span across page breaks.
    """
    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        # Fix OCR hyphens: across page breaks and within lines
        text = re.sub(r'(\w)-\s*\n+\s*(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Drop front matter before first question
        first_q = re.search(r'\n\n(?:Has |What |Who |How |Why |Does |Is |Are |By |Do |Can |If )', text)
        if first_q:
            text = text[first_q.start():]
        # Strip header paragraphs (lesson headers and short mostly-uppercase page titles)
        paragraphs = re.split(r'\n\n+', text)
        filtered = []
        for p in paragraphs:
            stripped = p.strip()
            if not stripped:
                continue
            if re.match(r'^LESSON\s+[IVX]+', stripped):
                continue
            alpha = sum(1 for c in stripped if c.isalpha())
            upper = sum(1 for c in stripped if c.isupper())
            if alpha > 0 and upper / alpha > 0.65 and len(stripped) < 60:
                continue
            filtered.append(stripped)

        pairs = []
        for p in filtered:
            if '?' not in p:
                continue
            idx = p.index('?')
            question = re.sub(r'\s+', ' ', p[:idx + 1].strip())
            answer = re.sub(r'\s+', ' ', p[idx + 1:].strip())
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class InvestorsExtractor(BaseExtractor):
    """
    The Investor's Catechism. Format: 'What is TERM ?' / answer, similar to FamiliarThings.
    Key terms appear in ALL CAPS in questions and are lowercased on extraction.
    Front matter (index) precedes the content; page headers are interspersed as noise.
    """
    _PAGE_HEADER = re.compile(r'\n\n[\w]+ THE INVESTOR\'S CATECHISM[\w ]*\n\n|\n\nTHE INVESTOR\'S CATECHISM[\w ]*\n\n')

    def extract_pairs(self, text):
        text = strip_gutenberg_boilerplate(text)
        # Drop everything before the first question
        first_q = re.search(r'\nWhat ', text)
        if first_q:
            text = text[first_q.start():]
        # Strip page headers
        text = self._PAGE_HEADER.sub('\n\n', text)
        # Normalize single-line Q&A: "What is X? Answer." → two paragraphs
        text = re.sub(r'^(What [^\n]+?\?)\s+([A-Z])', r'\1\n\n\2', text, flags=re.MULTILINE)

        pattern = re.compile(r'^(What [^\n]+\?)\n\n(.+?)(?=\nWhat |\Z)', re.MULTILINE | re.DOTALL)

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group().lower(), q_text.strip())
            question = re.sub(r" {2,}", " ", question)
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs


class AdvancedQuestionsExtractor(BaseExtractor):
    """
    Dime Question Books: Advanced series. Format: numbered questions ('N. question text')
    followed by 'Ans. — answer'. Answer marker has many OCR variants (An?., Aks,, Anb.,
    ANS.—). Note.— and Query.— paragraphs follow some answers and are excluded.
    Some Q+A pairs appear on a single paragraph with the Ans marker inline.
    Answers may span multiple paragraphs separated by page headers.
    """
    ANS_RE = re.compile(r'A[\w?]{1,3}[.,]?\s*[—\-]\s*')
    # Inline variant requires sentence-ending punct before the marker to avoid
    # false positives on hyphenated words like "(Am- phibia.)"
    INLINE_ANS_RE = re.compile(r'(?<=[.?!])\s+A[\w?]{1,3}[.,]?\s*[—\-]\s*')
    STOP_RE = re.compile(r'^(?:\d+\.|Note[.\-]|Query|Quer\s*y)', re.IGNORECASE)

    _PAGE_HEADER_RE = re.compile(
        r'^(?:'
        r'\d+\s+\w'             # "93 Dxmb Question Books,"
        r'|.*Question Books'    # "Dime Question Books."
        r'|\w+[\.\s]+[\dIVXivx]{1,5}\W*$'  # "Chemistry. 26&" / "Literature. IT"
        r')',
        re.IGNORECASE
    )

    def _is_page_header(self, p):
        lines = [l.strip() for l in p.split('\n') if l.strip()]
        if not lines or len(lines) > 2:
            return False
        return bool(self._PAGE_HEADER_RE.match(p)) or _is_symbological_header(p)

    def _collect_answer(self, paragraphs, start, first_text):
        """Collect answer text across paragraphs until a stop marker."""
        parts = [first_text]
        j = start
        while j < len(paragraphs):
            nxt = paragraphs[j].strip()
            if self.STOP_RE.match(nxt) or self.ANS_RE.match(nxt):
                break
            if not self._is_page_header(nxt):
                parts.append(nxt)
            j += 1
        return re.sub(r'\s+', ' ', ' '.join(parts)).strip()

    def extract_pairs(self, text):
        text = re.sub(r'(\w)-[ ]*\n[ ]*(\w)', r'\1\2', text)

        paragraphs = re.split(r'\n\n+', text)
        pairs = []
        i = 0
        while i < len(paragraphs):
            p = paragraphs[i].strip()
            q_match = re.match(r'^(\d+\.\s+)(.+)', p, re.DOTALL)
            if q_match:
                body = q_match.group(2)
                inline = self.INLINE_ANS_RE.search(body)
                if inline:
                    # Q and Ans on same paragraph — split inline
                    q = re.sub(r'\s+', ' ', body[:inline.start()]).strip()
                    a = self._collect_answer(paragraphs, i + 1, body[inline.end():])
                    if q and a:
                        pairs.append({"q": q, "a": a})
                else:
                    q = re.sub(r'\s+', ' ', body).strip()
                    # Look ahead for the answer paragraph
                    j = i + 1
                    while j < len(paragraphs):
                        candidate = paragraphs[j].strip()
                        ans_match = self.ANS_RE.match(candidate)
                        if ans_match:
                            a = self._collect_answer(paragraphs, j + 1, candidate[ans_match.end():])
                            if q and a:
                                pairs.append({"q": q, "a": a})
                            break
                        if re.match(r'^\d+\.', candidate):
                            break
                        j += 1
            i += 1
        return pairs


class WorldHistoryExtractor(BaseExtractor):
    """
    Catechism of Universal History. Format: 'N. Q. question' / 'A. answer'.
    Q. sometimes OCRs as Q,. or Q,.. Single-line items ('N. Q. question? A. answer.')
    also occur. Standalone page numbers and CHAP. headers are interspersed as noise.
    """
    def extract_pairs(self, text):
        # Fix OCR hyphens
        text = re.sub(r'(\w)- *\n *(\w)', r'\1\2', text)
        text = re.sub(r'(\w)- +(\w)', r'\1\2', text)
        # Strip standalone page numbers and short OCR artifacts between blank lines
        text = re.sub(r'\n\n\d{1,3}\n\n', '\n\n', text)
        text = re.sub(r'\n\n[a-zA-Z]{1,2}\n\n', '\n\n', text)
        # Strip CHAP. header lines
        text = re.sub(r'\n\nCHAP\.[^\n]*\n\n', '\n\n', text)
        # Normalize 'A,' (OCR for 'A.') at line-start
        text = re.sub(r'^A,\s', 'A. ', text, flags=re.MULTILINE)
        # Strip inline noise tokens that appear before Q-markers (page refs, abbrevs)
        # e.g. 'p €2. Q.' (inline), 'bb» Q.' (line-start with non-alpha prefix)
        text = re.sub(r'[^\w\n]+(\d+\.\s+Q[.,]+\s)', r'\n\n\1', text)
        # Line-start Q. with non-alpha prefix (bb» Q., Sr. Q., p €2. Q.)
        text = re.sub(r'^[^A-Za-z0-9\n]*\s*Q\.\s', 'Q. ', text, flags=re.MULTILINE)
        text = re.sub(r'^[A-Za-z]{1,3}[^\w\s]*\s*Q\.\s', 'Q. ', text, flags=re.MULTILINE)
        # Normalize single-line Q&A: "N. Q. question? A. answer" → two paragraphs
        text = re.sub(r'^(\d+\.\s+Q[.,]+\s+.+?\?)\s+(A[.,]\s+)', r'\1\n\nA. ', text, flags=re.MULTILINE)
        # Inline Q. split (for answers that absorbed subsequent Q markers)
        text = re.sub(r'([.,?])\s*(Q\. )', r'\1\n\n\2', text)

        pattern = re.compile(
            r'^(?:\d+\.\s+)?Q[.,]+\s+(.+?)\n\nA\.\s+(.+?)(?=\n\n(?:\d+\.\s+)?Q[.,]|\Z)',
            re.MULTILINE | re.DOTALL
        )

        pairs = []
        for q_text, a_text in pattern.findall(text):
            question = re.sub(r" {2,}", " ", q_text.strip().replace("\n", " "))
            answer = re.sub(r" {2,}", " ", a_text.strip().replace("\n", " "))
            if question and answer:
                pairs.append({"q": question, "a": answer})

        return pairs
