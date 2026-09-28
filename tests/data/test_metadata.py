from ads_rank_lab.data.metadata import preferred_release_names


def test_preferred_release_names_match_research_plan() -> None:
    assert preferred_release_names() == ("Season 2", "Season 3")
