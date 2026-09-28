"""Dataset metadata used to keep season assumptions explicit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRelease:
    """Metadata about an intended iPinYou release/season."""

    name: str
    preferred: bool
    notes: str


IPINYOU_RELEASES: tuple[DatasetRelease, ...] = (
    DatasetRelease(
        name="Season 2",
        preferred=True,
        notes="Preferred for AdsRankLab when advertiser/context fields needed by the study are available.",
    ),
    DatasetRelease(
        name="Season 3",
        preferred=True,
        notes="Preferred for AdsRankLab when advertiser/context fields needed by the study are available.",
    ),
)


def preferred_release_names() -> tuple[str, ...]:
    """Return release names currently preferred by the research plan."""
    return tuple(release.name for release in IPINYOU_RELEASES if release.preferred)
