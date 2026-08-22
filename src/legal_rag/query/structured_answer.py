"""Strict structured-output contract for model-generated legal answers."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GeneratedAnswer(BaseModel):
    """Small schema suitable for reliable local-model generation."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    evidence_ids: list[str]
    insufficient_evidence: bool = False

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must be unique")
        return values
