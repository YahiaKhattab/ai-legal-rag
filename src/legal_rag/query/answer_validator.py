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
        [٠-٩]+(?:[٫،.][٠-٩]+)*
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
}


_MILLION_PATTERN = re.compile(
    r"""
    (
        [0-9]+(?:[.,][0-9]+)*
        |
        [٠-٩]+(?:[٫،.][٠-٩]+)*
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
    """Normalize Arabic and English digits."""

    translation = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩",
        "0123456789",
    )

    value = value.translate(translation)

    value = value.replace("٫", ".")
    value = value.replace("،", ",")

    return value.replace(",", "").strip(".")


def extract_numbers(text: str) -> set[str]:
    """Extract numeric digit values from text."""

    numbers: set[str] = set()

    for match in _NUMBER_PATTERN.findall(text):
        normalized = _normalize_number(match)

        if normalized:
            numbers.add(normalized)

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
        rf"(?:تقل|يقل)\s*(?:قيمتها|قيمته|القيمة)?\s*(?:عن|على)?\s*{number}\s*(?:مليون|ملايين|million|millions)",
        rf"(?:لا تجاوز|لا يجاوز|لا تتجاوز|لا يتجاوز)"
        rf"\s*(?:قيمتها|قيمته|القيمة)?\s*{number}"
        rf"\s*(?:مليون|ملايين|million|millions)",
        rf"(?:تزيد|يزيد)\s*(?:قيمتها|قيمته|القيمة)?\s*(?:على|عن)\s*{number}\s*(?:مليون|ملايين|million|millions)",
        rf"(?:تجاوز|تجاوزت|يتجاوز|تتجاوز)\s*(?:قيمتها|قيمته|القيمة)?\s*{number}\s*(?:مليون|ملايين|million|millions)",
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


def validate_numeric_claims(
    query: str,
    answer: str,
    evidence_text: str,
) -> tuple[bool, set[str], set[str]]:
    """Validate numeric and legal-threshold claims.

    Rules:

    1. Numbers supplied by the user may appear in the answer.
    2. Numbers explicitly present in the evidence may appear in the answer.
    3. A legal threshold appearing in the answer must agree with the
       corresponding threshold found in the evidence.
    4. A user case amount must NOT replace a different legal threshold.

    Returns:
        (
            is_valid,
            unsupported_or_conflicting_values,
            evidence_numeric_values,
        )
    """

    query_numbers = extract_numbers(query)
    answer_numbers = extract_numbers(answer)
    evidence_numbers = extract_numbers(evidence_text)

    unsupported_numbers = answer_numbers - query_numbers - evidence_numbers

    query_millions = _extract_million_values(query)

    answer_millions = _extract_million_values(answer)

    evidence_millions = _extract_million_values(evidence_text)

    answer_thresholds = _extract_threshold_values(answer)

    evidence_thresholds = _extract_threshold_values(evidence_text)

    # ---------------------------------------------------------
    # Rule 1:
    # A generated legal threshold must exist in the evidence.
    # ---------------------------------------------------------
    for value in answer_thresholds:
        if value not in evidence_thresholds:
            unsupported_numbers.add(value)

    # ---------------------------------------------------------
    # Rule 2:
    # If the evidence contains a legal threshold and the answer
    # contains another threshold, the answer is invalid.
    #
    # Example:
    #
    # Evidence:
    #     أقل من خمسة ملايين
    #
    # Answer:
    #     أقل من ثلاثة ملايين
    #
    # Even though "3 million" exists in the user question,
    # it is invalid when used as a legal threshold.
    # ---------------------------------------------------------
    if (
        evidence_thresholds
        and answer_thresholds
        and not answer_thresholds.issubset(evidence_thresholds)
    ):
        unsupported_numbers.update(answer_thresholds - evidence_thresholds)

    # ---------------------------------------------------------
    # Rule 3:
    # A million value that is merely a case amount is allowed.
    #
    # Example:
    #     "قضيتك البالغة 3 ملايين..."
    #
    # if 3 million exists in the question.
    # ---------------------------------------------------------
    for value in answer_millions:
        if value in query_millions:
            continue

        if value in evidence_millions:
            continue

        # If it is neither a user value nor an evidence value,
        # it is unsupported.
        unsupported_numbers.add(value)

    # ---------------------------------------------------------
    # Rule 4:
    # Preserve evidence thresholds even when the user supplied
    # another amount.
    #
    # Example:
    #
    # Query:     3 million
    # Evidence:  5 million threshold
    # Answer:    3 million threshold
    #
    # -> invalid.
    # ---------------------------------------------------------
    if evidence_thresholds:
        for answer_value in answer_thresholds:
            if answer_value not in evidence_thresholds:
                unsupported_numbers.add(answer_value)

    evidence_numeric_values = evidence_numbers | evidence_millions | evidence_thresholds

    return (
        not unsupported_numbers,
        unsupported_numbers,
        evidence_numeric_values,
    )
