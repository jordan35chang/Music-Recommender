# Music-Recommender

## **Problem**
Music streaming platforms need to recommend tracks that fit a given playlist, even when the playlist is short or the tracks in it are unfamiliar to the system. This project builds a playlist-based track recommender: given a playlist's existing tracks, predict which other tracks are likely to belong in it. A particular focus is handling the "cold-start" problem, making reasonable recommendations for playlists or tracks the model hasn't seen much of before.

## **Approach**
The project is build in stages, each with its own evaluation:  
1. **Baseline**: a simple popularity-based recommender (e.g. most popular tracks overall, or most popular within similar playlists) to establish a floor and validate the evaluation pipelline.
2. **Collaborative filtering model**: Weighted Regularized Matric Factorization (WRMF) trained on playlist-track interation data, using the `implicit` library
3. **Hybrid extension (*planned*)**: incorporating content-based features (audio features, genre, artist embeddings) alongside collaborative signals to improve cold-start performance for new or lesser-known tracks.
  

Each stage is evaluated using precision@k, recall@k, and NDCG rather than plain accuracy metrics, since these better reflect ranked-recommendation quality.  

## **Data**
This project uses the <u>Spotify Million Playlist Dataset</u>, which contains 1 million user-created playhlists and their associated tracks. The dataset is not included in this repository (see `.gitignore`): download instructions are in the section below.  

## **Results**
*TBD*  

## **How to run**
*TBD*