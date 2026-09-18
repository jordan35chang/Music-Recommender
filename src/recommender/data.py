import json 
import glob
import os
import pandas as pd

def load_slice(filepath):
    """
    Load a single MPD slice JSON file.   
    Args:
        filepath: path to a slice file
    
    Returns:
        dict with keys 'info' and 'playlists'
    """
    with open(filepath) as f:
        return json.load(f)
    
def flat_interactions(playlists):
    """
    Flatten a list of playlist records into (playlist, track)
    Args:
        playlists: list of playlist dicts, found under data['playlists']
        
    Returns:
        pd.DataFrame with columnns pid, track_uri, artist_name, track_name
    """
    rows = []
    for playlist in playlists:
        pid = playlist["pid"]
        for track in playlist["tracks"]:
            rows.append({
                "pid": pid,
                "track_uri": track["track_uri"],
                "artist_name": track["artist_name"],
                "track_name": track["track_name"],
            })
    return pd.DataFrame(rows)

def load_dataset(data_dir, num_slices=None):
    """
    load and flatten multiple MPD slice files from a directory
    Args:
        data_dir: path to the folder containing mpd.slice.*.json files
        num_slices: if provided, only load this many slice files, if None, 
        load all slices found
        
    Returns: 
        pd.DataFrame with columns: pid, track_uri, artist_name, track_name,
        combine interactions from every loaded slice
    """
    slice_paths = sorted(glob.glob(os.path.join(data_dir, "mpd.slice.*.json")))
    
    if not slice_paths:
        raise FileNotFoundError(f"No slice files found in {data_dir}")
    
    if num_slices is not None:
        slice_paths = slice_paths[:num_slices]
    all_dfs = []
    for path in slice_paths:
        data = load_slice(path)
        df = flat_interactions(data["playlists"])
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)