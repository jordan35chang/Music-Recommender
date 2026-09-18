"""
Tests for src/recommender/data.py
"""

import json
import pytest
from src.recommender.data import load_slice, flat_interactions, load_dataset

def make_fake_playlist(pid, track_count=5):
    """
    Helper to build a fake playlist dict matching the real MPD structure
    """
    return {
        "pid": pid,
        "name": f"Test Playlist {pid}",
        "tracks": [
            {
                "pos": i,
                "track_uri": f"spotify:track:fake{pid}_{i}",
                "artist_name": f"Artist {pid}_{i}",
                "track_name": f"Track {pid}_{i}"
            }
            for i in range(track_count)
        ],
    }

def make_fake_slice_file(path, num_playlists=2, tracks_per_playlist=3):
    """
    Write a small fake MPD slice JSON file to the given path
    """
    fake_data = {
        "info": {"generated_on":  "2026-01-01", "slice": "0-1", "version": "v1"},
        "playlists": [
            make_fake_playlist(pid, tracks_per_playlist)
            for pid in range(num_playlists)
        ]
    }
    with open(path, "w") as f:
        json.dump(fake_data, f)
    return fake_data

# load_slice tests
def test_load_slice_returns_expected_keys(tmp_path):
    filepath = tmp_path / "mpd.slice.0-1.json"
    make_fake_slice_file(filepath, num_playlists=2)

    result = load_slice(filepath)

    assert "info" in result
    assert "playlists" in result
    assert len(result["playlists"]) == 2

# flat_interactions tests
def test_flat_interactions_shape_columns():
    playlists = [make_fake_playlist(pid=0, track_count=3)]

    df = flat_interactions(playlists)

    assert df.shape[0] == 3
    assert list(df.columns) == ["pid", "track_uri", "artist_name", "track_name"]

def test_flat_interactions_multiple_playlists():
    playlists = [
        make_fake_playlist(pid=0, track_count=2),
        make_fake_playlist(pid=1, track_count=4)
    ]

    df = flat_interactions(playlists)

    assert df.shape[0] == 6 # 2 + 4 tracks
    assert set(df["pid"].unique()) == {0, 1}

def test_load_dataset_concatenates_multiple_slices(tmp_path):
    make_fake_slice_file(tmp_path / "mpd.slice.0-1.json", num_playlists=2, tracks_per_playlist=3)
    make_fake_slice_file(tmp_path / "mpd.slice.2-3.json", num_playlists=2, tracks_per_playlist=3)

    df = load_dataset(str(tmp_path))

    assert df.shape[0] == 12 # 2 files * 2 playlists * 3 tracks

def test_load_dataset_respects_num_slices(tmp_path):
    make_fake_slice_file(tmp_path / "mpd.slice.0-1.json", num_playlists=2, tracks_per_playlist=3)
    make_fake_slice_file(tmp_path / "mpd.slice.2-3.json", num_playlists=2, tracks_per_playlist=3)

    df = load_dataset(str(tmp_path), num_slices=1)

    assert df.shape[0] == 6 # only the first file's 2 * 3 tracks

def test_load_dataset_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path))