"""
Tests for src/recommender/model.py
"""

import pandas as pd
import pytest
from src.recommender.model import PopularityBaseline

def make_fake_interactions():
    """
    Build a small fake interactions dataframe with known popularity counts
    
    track_A appears in 3 playlists
    track_B appears in 2 playlists
    track_C appears in 1 playlist
    """
    rows = [
        {"pid": 0, "track_uri": "track_A", "artist_name": "Artist A", "track_name": "Song A"},
        {"pid": 1, "track_uri": "track_A", "artist_name": "Artist A", "track_name": "Song A"},
        {"pid": 2, "track_uri": "track_A", "artist_name": "Artist A", "track_name": "Song A"},
        {"pid": 0, "track_uri": "track_B", "artist_name": "Artist B", "track_name": "Song B"},
        {"pid": 1, "track_uri": "track_B", "artist_name": "Artist B", "track_name": "Song B"},
        {"pid": 3, "track_uri": "track_C", "artist_name": "Artist C", "track_name": "Song C"},
    ]
    return pd.DataFrame(rows)

def test_fit_ranks_by_popularity():
    interactions = make_fake_interactions()
    model = PopularityBaseline().fit(interactions)

    ranking = model.popularity_ranking

    assert list(ranking.index) == ["track_A", "track_B", "track_C"]
    assert ranking["track_A"] == 3
    assert ranking["track_B"] == 2
    assert ranking["track_C"] == 1

def test_recommend_excludes_existing_tracks():
    interactions = make_fake_interactions()
    model = PopularityBaseline().fit(interactions)

    # playlist 0 already has track_A and track_B
    recs = model.recommend(pid=0, interactions=interactions, k=5)

    assert "track_A" not in recs
    assert "track_B" not in recs
    assert recs[0] == "track_C"

def test_recommend_respects_k():
    interactions = make_fake_interactions()
    model = PopularityBaseline().fit(interactions)

    # playlist 3 only has track_C, so track_A and track_B are both available
    recs = model.recommend(pid=3, interactions=interactions, k=1)

    assert len(recs) == 1
    assert recs[0] == "track_A"

def test_recommend_raises_if_not_fitted():
    interactions = make_fake_interactions()
    model = PopularityBaseline()

    with pytest.raises(RuntimeError):
        model.recommend(pid=0, interactions=interactions, k=5)

def test_recommend_with_names_returns_readable_tuples():
    interactions = make_fake_interactions()
    model = PopularityBaseline().fit(interactions)

    recs = model.recommend_with_names(pid=3, interactions=interactions, k=1)

    assert recs == [("track_A", "Artist A", "Song A")]