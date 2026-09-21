"""Deterministic validation for generated legal answers.

The validator protects legal numeric rules from being changed by the LLM.

Example:

    User question:
        "لو القضية بـ3 مليون جنيه..."

    Legal evidence:
        "الدعاوى التي تقل قيمتها عن خمسة ملايين جنيه..."

    Valid:
        "قضيتك البالغة 3 ملايين تقع ضمن اختصاص الدائرة الابتدائية."

    Invalid:
        "الدعاوى التي تقل قيمتها عن 3 ملايين..."

The important distinction is between:
    - a user-provided case value
    - a legal threshold stated by the evidence
"""

from __future__ import annotations

import re


_NUMBER_PATTERN = re.compile(
    r"""
    (?:
        [0-9]+(?:[.,][0-9]+)*
        |
        [٠-٩۰-۹]+(?:[٫،.][٠-٩۰-۹]+)*
    )
    """,
    re.VERBOSE,
)


_ARABIC_NUMBER_WORDS = {
    "صفر": 0,
    "واحد": 1,
    "واحدة": 1,
    "اثنان": 2,
    "اثنين": 2,
    "اثنتان": 2,
    "اثنتين": 2,
    "ثلاثة": 3,
    "ثلاث": 3,
    "أربعة": 4,
    "أربع": 4,
    "خمسة": 5,
    "خمس": 5,
    "ستة": 6,
    "ست": 6,
    "سبعة": 7,
    "سبع": 7,
    "ثمانية": 8,
    "ثمان": 8,
    "تسعة": 9,
    "تسع": 9,
    "عشرة": 10,
    "عشر": 10,
    "عشره": 10,
    "عشرين": 20,
    "عشرون": 20,
    "ثلاثين": 30,
    "ثلاثون": 30,
    "أربعين": 40,
    "اربعين": 40,
    "أربعون": 40,
    "اربعون": 40,
    "خمسين": 50,
    "خمسون": 50,
    "ستين": 60,
    "ستون": 60,
    "سبعين": 70,
    "سبعون": 70,
    "ثمانين": 80,
    "ثمانون": 80,
    "تسعين": 90,
    "تسعون": 90,
}


_MILLION_PATTERN = re.compile(
    r"""
    (
        [0-9]+(?:[.,][0-9]+)*
        |
        [٠-٩۰-۹]+(?:[٫،.][٠-٩۰-۹]+)*
    )
    \s*
    (?:مليون|ملايين|مليونًا|مليوناً|million|millions)
    """,
    re.IGNORECASE | re.VERBOSE,
)


_ARABIC_MILLION_WORD_PATTERN = re.compile(
    r"""
    (
        صفر
        |واحد
        |واحدة
        |اثنان
        |اثنين
        |اثنتان
        |اثنتين
        |ثلاثة
        |ثلاث
        |أربعة
        |أربع
        |خمسة
        |خمس
        |ستة
        |ست
        |سبعة
        |سبع
        |ثمانية
        |ثمان
        |تسعة
        |تسع
        |عشرة
    )
    \s*
    (?:مليون|ملايين|مليونًا|مليوناً)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_number(value: str) -> str:
    """Normalize Arabic, Persian, and English digits."""

    translation = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789",
    )

    value = value.translate(translation)

    value = value.replace("٫", ".")
    value = value.replace("،", ",")

    return value.replace(",", "").strip(".")



_ARABIC_SMALL = {
    "صفر": 0, "واحد": 1, "واحدة": 1, "احد": 1, "احدى": 1,
    "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2,
    "ثلاثة": 3, "ثلاث": 3, "اربعة": 4, "اربع": 4,
    "خمسة": 5, "خمس": 5, "ستة": 6, "ست": 6,
    "سبعة": 7, "سبع": 7, "ثمانية": 8, "ثمان": 8,
    "تسعة": 9, "تسع": 9, "عشرة": 10, "عشر": 10,
}
_ARABIC_TENS = {
    "عشرين": 20, "عشرون": 20, "ثلاثين": 30, "ثلاثون": 30,
    "اربعين": 40, "اربعون": 40, "خمسين": 50, "خمسون": 50,
    "ستين": 60, "ستون": 60, "سبعين": 70, "سبعون": 70,
    "ثمانين": 80, "ثمانون": 80, "تسعين": 90, "تسعون": 90,
}
_ARABIC_HUNDREDS = {
    "مئة": 100, "مائه": 100, "مائة": 100, "مئتان": 200, "مائتان": 200,
    "مئتين": 200, "مائتين": 200, "ثلاثمئة": 300, "ثلاثمائة": 300,
    "اربعمئة": 400, "اربعمائة": 400, "خمسمئة": 500, "خمسمائة": 500,
    "ستمئة": 600, "ستمائة": 600, "سبعمئة": 700, "سبعمائة": 700,
    "ثمانمئة": 800, "ثمانمائة": 800, "تسعمئة": 900, "تسعمائة": 900,
}
def _norm_ar_word(w: str) -> str:
    w = re.sub(r"[ًٌٍَُِّْٰ]", "", w).replace("ـ", "")
    w = w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return w

def _parse_arabic_integer_words(words: list[str]) -> int | None:
    """Parse a consecutive Arabic cardinal-number phrase (up to millions)."""
    vals = []
    for raw in words:
        w = _norm_ar_word(raw)
        if w.startswith("و") and w[1:] in (_ARABIC_SMALL | _ARABIC_TENS | _ARABIC_HUNDREDS):
            w = w[1:]
        if w in _ARABIC_SMALL: vals.append(("n", _ARABIC_SMALL[w]))
        elif w in _ARABIC_TENS: vals.append(("n", _ARABIC_TENS[w]))
        elif w in _ARABIC_HUNDREDS: vals.append(("n", _ARABIC_HUNDREDS[w]))
        elif w in ("الف", "الاف", "الفا", "الفين"): vals.append(("scale", 1000))
        elif w in ("مليون", "ملايين"): vals.append(("scale", 1000000))
        elif w == "و": continue
        else: return None
    if not vals: return None
    total = current = 0
    for typ, val in vals:
        if typ == "scale":
            total += max(current, 1) * val
            current = 0
        else:
            current += val
    return total + current

def _extract_arabic_integer_phrases(text: str) -> set[int]:
    tokens = re.findall(r"[أإآء-ي]+", text.lower())
    out: set[int] = set()
    run: list[str] = []
    for token in tokens + ["__END__"]:
        norm = _norm_ar_word(token) if token != "__END__" else token
        recognized = (norm in _ARABIC_SMALL or norm in _ARABIC_TENS or
                      norm in _ARABIC_HUNDREDS or norm in
                      {"الف", "الاف", "الفا", "الفين", "مليون", "ملايين", "و"} or
                      (norm.startswith("و") and norm[1:] in
                       (_ARABIC_SMALL | _ARABIC_TENS | _ARABIC_HUNDREDS)))
        if recognized:
            run.append(token)
        else:
            if run:
                value = _parse_arabic_integer_words(run)
                if value is not None: out.add(value)
                run = []
    if run:
        value = _parse_arabic_integer_words(run)
        if value is not None: out.add(value)
    return out

def extract_numbers(text: str) -> set[str]:
    """Extract numeric digit values from text."""

    numbers: set[str] = set()

    for match in _NUMBER_PATTERN.findall(text):
        normalized = _normalize_number(match)

        if normalized:
            numbers.add(normalized)

    numbers.update(str(n) for n in _extract_arabic_integer_phrases(text))
    return numbers


def extract_arabic_number_words(text: str) -> set[int]:
    """Extract simple Arabic number words."""

    words = re.findall(
        r"[أإآء-ي]+",
        text,
    )

    numbers: set[int] = set()

    for word in words:
        normalized = word.strip()

        if normalized in _ARABIC_NUMBER_WORDS:
            numbers.add(_ARABIC_NUMBER_WORDS[normalized])

    return numbers


def _extract_million_values(text: str) -> set[str]:
    """Extract explicit monetary million values.

    Examples:
        3 million
        5 million
        ٣ ملايين
        خمسة ملايين

    Returned format:
        {"3 million", "5 million"}
    """

    values: set[str] = set()

    # Numeric forms:
    # 3 million
    # 5 ملايين
    # ٣ ملايين
    # ۳ میلیون
    for match in _MILLION_PATTERN.finditer(text):
        number = _normalize_number(match.group(1))

        if number:
            values.add(f"{number} million")

    # Arabic word forms:
    # خمسة ملايين
    for match in _ARABIC_MILLION_WORD_PATTERN.finditer(text):
        word = match.group(1).strip()

        number_word_value = _ARABIC_NUMBER_WORDS.get(word)

        if number_word_value is not None:
            values.add(f"{number_word_value} million")

    return values


def _million_number(value: str) -> float:
    """Convert '5 million' into 5.0."""

    number = value.replace(
        " million",
        "",
    )

    return float(number)


def _million_base_numbers(values: set[str]) -> set[str]:
    """Return the numeric base of validated million values.

    Example:
        {"5 million", "3 million"} -> {"5", "3"}

    This is used only to prevent a validated million value from being
    incorrectly rejected because its numeric component was separately
    extracted by ``extract_numbers``.
    """

    return {
        value.removesuffix(" million").strip()
        for value in values
    }


def _contains_legal_threshold(
    text: str,
    million_value: str,
) -> bool:
    """Check whether a million value is used as a legal threshold.

    This intentionally focuses on common legal threshold language instead
    of trying to understand the entire generated answer.
    """

    number = re.escape(
        million_value.replace(
            " million",
            "",
        )
    )

    threshold_patterns = [
        rf"(?:أقل|اقل|أقل من|اقل من)\s*(?:قيمة\s*)?"
        rf"(?:الدعوى|الدعاوى|المنازعة|المنازعات)?\s*{number}"
        rf"\s*(?:مليون|ملايين|million|millions)",

        rf"(?:تقل|يقل)\s*(?:قيمتها|قيمته|القيمة)?\s*(?:عن|على)?\s*{number}\s*"
        rf"(?:مليون|ملايين|million|millions)",

        rf"(?:لا تجاوز|لا يجاوز|لا تتجاوز|لا يتجاوز)"
        rf"\s*(?:قيمتها|قيمته|القيمة)?\s*{number}"
        rf"\s*(?:مليون|ملايين|million|millions)",

        rf"(?:تزيد|يزيد)\s*(?:قيمتها|قيمته|القيمة)?\s*(?:على|عن)\s*{number}\s*"
        rf"(?:مليون|ملايين|million|millions)",

        rf"(?:تجاوز|تجاوزت|يتجاوز|تتجاوز)\s*(?:قيمتها|قيمته|القيمة)?\s*{number}\s*"
        rf"(?:مليون|ملايين|million|millions)",
    ]

    normalized = text.lower()

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in threshold_patterns
    )


def _extract_threshold_values(text: str) -> set[str]:
    """Return million values that appear to be legal thresholds."""

    values = _extract_million_values(text)

    return {
        value
        for value in values
        if _contains_legal_threshold(
            text,
            value,
        )
    }


_UNIT_ALIASES = {
    "يوم": "day",
    "يوما": "day",
    "يومًا": "day",
    "يوماً": "day",
    "يومين": "day",
    "أيام": "day",
    "ايام": "day",
    "سنة": "year",
    "سنوات": "year",
    "عام": "year",
    "أعوام": "year",
    "اعوام": "year",
    "شهر": "month",
    "شهور": "month",
    "أشهر": "month",
    "اشهر": "month",
    "ساعة": "hour",
    "ساعات": "hour",
    "أسبوع": "week",
    "اسبوع": "week",
    "أسابيع": "week",
    "اسابيع": "week",
    "دقيقة": "minute",
    "دقائق": "minute",
    "day": "day",
    "days": "day",
    "year": "year",
    "years": "year",
    "month": "month",
    "months": "month",
    "hour": "hour",
    "hours": "hour",
    "week": "week",
    "weeks": "week",
    "minute": "minute",
    "minutes": "minute",
}


def _extract_number_unit_claims(text: str) -> set[tuple[int, str]]:
    """Extract simple number + duration-unit claims, including Arabic number words."""

    tokens = re.findall(
        r"[0-9٠-٩۰-۹]+|[أإآء-ي]+|[A-Za-z]+",
        text.lower(),
    )

    claims: set[tuple[int, str]] = set()

    # Strip common Arabic case endings/diacritics for reliable unit matching.
    def norm_token(token: str) -> str:
        token = token.replace("ـ", "")
        token = re.sub(r"[ًٌٍَُِّْٰ]", "", token)
        return token

    normalized_tokens = [norm_token(t) for t in tokens]

    for i, token in enumerate(normalized_tokens[:-1]):
        value: int | None = None

        if token.isdigit() or re.fullmatch(r"[٠-٩۰-۹]+", token):
            try:
                value = int(_normalize_number(token))
            except ValueError:
                pass
        else:
            value = _ARABIC_NUMBER_WORDS.get(token)

        unit = _UNIT_ALIASES.get(normalized_tokens[i + 1])

        if unit:
            if value is not None:
                claims.add((value, unit))
            # Parse a multi-word cardinal immediately before the unit,
            # e.g. "واحد وعشرين يوماً" -> (21, day).
            start = i
            while start > 0 and (
                normalized_tokens[start - 1] in (_ARABIC_SMALL | _ARABIC_TENS | _ARABIC_HUNDREDS)
                or normalized_tokens[start - 1] == "و"
                or (normalized_tokens[start - 1].startswith("و") and
                    normalized_tokens[start - 1][1:] in (_ARABIC_SMALL | _ARABIC_TENS | _ARABIC_HUNDREDS))
            ):
                start -= 1
            phrase = normalized_tokens[start:i + 1]
            parsed = _parse_arabic_integer_words(phrase)
            if parsed is not None:
                claims.add((parsed, unit))

    return claims


def _strip_list_numbering(text: str) -> str:
    """Remove list labels (1., 1-, ١), etc.) before numeric-claim checks.

    List labels are formatting, not substantive legal numeric claims.
    Numbers elsewhere in the answer remain subject to validation.
    """

    return re.sub(
        r"(?m)^\s*(?:[-*•]\s*)?[0-9٠-٩۰-۹]+[.)、-]\s*",
        "",
        text,
    )


def validate_numeric_claims(
    query: str,
    answer: str,
    evidence_text: str,
) -> tuple[bool, set[str], set[str]]:
    """Validate numeric claims, legal thresholds, and number-duration units.

    A number followed by a duration unit must be supported as the same
    number-unit pair by either the question or the evidence.
    """

    query_numbers = extract_numbers(query)

    # Ignore numeric labels used only to enumerate list items.
    answer_for_numeric_checks = _strip_list_numbering(answer)

    answer_numbers = extract_numbers(answer_for_numeric_checks)
    evidence_numbers = extract_numbers(evidence_text)

    query_millions = _extract_million_values(query)
    answer_millions = _extract_million_values(answer_for_numeric_checks)
    evidence_millions = _extract_million_values(evidence_text)

    validated_million_values = query_millions | evidence_millions
    validated_million_base_numbers = _million_base_numbers(
        validated_million_values
    )

    unsupported_numbers = (
        answer_numbers
        - query_numbers
        - evidence_numbers
        - validated_million_base_numbers
    )

    answer_thresholds = _extract_threshold_values(
        answer_for_numeric_checks
    )
    evidence_thresholds = _extract_threshold_values(evidence_text)

    for value in answer_thresholds:
        if value not in evidence_thresholds:
            unsupported_numbers.add(value)

    if evidence_thresholds and answer_thresholds:
        unsupported_numbers.update(
            answer_thresholds - evidence_thresholds
        )

    for value in answer_millions:
        if value not in query_millions and value not in evidence_millions:
            unsupported_numbers.add(value)

    if evidence_thresholds:
        unsupported_numbers.update(
            answer_thresholds - evidence_thresholds
        )

    # Reject changed or hallucinated duration units, e.g. evidence "10 days"
    # but generated answer "10 years" or "ten ages".
    supported_duration_claims = (
        _extract_number_unit_claims(query)
        | _extract_number_unit_claims(evidence_text)
    )

    for value, unit in _extract_number_unit_claims(
        answer_for_numeric_checks
    ):
        if (value, unit) not in supported_duration_claims:
            unsupported_numbers.add(f"{value} {unit}")

    evidence_numeric_values = (
        evidence_numbers
        | evidence_millions
        | evidence_thresholds
    )

    return (
        not unsupported_numbers,
        unsupported_numbers,
        evidence_numeric_values,
    )