"""
Baseline recommender model for the music recommender

implements a simple popularity-based baseline: recommends the most
popular tracks overall, excluding tracks already in the target playlist.
serves as a floor to beat with more sophisticated models
"""

import pandas as pd

class PopularityBaseline:
    """
    Recommends the most popular tracks, exvluding ones already in the playlist
    """
    def __init__(self):
        self.popularity_ranking = None # pd.Series: track_uri -> count, sorted desc
        self.track_lookup = None # dict: track_uri -> (artist_name, track_name)

    def fit(self, interactions):
        """
        Compute track popularity from training interaction data
        
        Args:
            interactions: pd.DataFrame with columns pid, track_uri,
            artist_name, track_name
        """
        self.popularity_ranking = (
            interactions["track_uri"].value_counts()
        )

        # keep one artist/track name per track_uri for readable output
        self.track_lookup = (
            interactions.drop_duplicates(subset="track_uri")
            .set_index("track_uri")[["artist_name", "track_name"]]
            .to_dict(orient="index")
        )

        return self
    
    def recommend(self, pid, interactions, k=10):
        """
        Recommend the top k most popular tracks not in playlist
        
        Args:
            pid: the playlist id to recommend for
            interactions: same intercations dataframe used elsewhere,
                          used to look up which tracks this playlist already has
            k: number of recommendations to return
            
        Returns:
            list of track_uri strings, most popular first
        """
        if self.popularity_ranking is None:
            raise RuntimeError("Model has not been fit yet")
        
        existing_tracks = set(
            interactions.loc[interactions["pid"] == pid, "track_uri"]
        )

        recommendations = []
        for track_uri in self.popularity_ranking.index:
            if track_uri not in existing_tracks:
                recommendations.append(track_uri)
            if len(recommendations) == k:
                break
        
        return recommendations
    
    def recommend_with_names(self, pid, interactions, k=10):
        """
        Same as recommend(), but returns (track_uri, artist_name, track_name) tuples
        """
        track_uris = self.recommend(pid, interactions, k=k)
        return [
            (uri, self.track_lookup[uri]["artist_name"], self.track_lookup[uri]["track_name"])
            for uri in track_uris
        ]