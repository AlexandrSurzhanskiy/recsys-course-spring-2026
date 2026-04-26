import json

import numpy as np

from .recommender import Recommender


MAX_HISTORY = 10


class TwoTowerRecommender(Recommender):
    def __init__(
        self,
        listen_history_redis,
        fallback_recommender,
        model_path,
        logger=None,
    ):
        self.listen_history_redis = listen_history_redis
        self.fallback_recommender = fallback_recommender
        self.available = False

        try:
            model = np.load(model_path)
            self.neighbors = model["neighbors"].astype(np.int64)
            self.popular_tracks = [int(track) for track in model["popular_tracks"]]
            self.available = True
        except (FileNotFoundError, KeyError):
            if logger is not None:
                logger.warning(
                    "Two-tower artifact not found or incompatible: %s. Falling back to Random.",
                    model_path,
                )

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if not self.available:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        history = self._load_user_history(user)
        if not history:
            return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)

        seen_tracks = {track for track, _ in history}
        for anchor in self._rank_anchors(history):
            for candidate in self.neighbors[anchor]:
                candidate = int(candidate)
                if candidate not in seen_tracks:
                    return candidate

        return self._recommend_popular(seen_tracks, user, prev_track, prev_track_time)

    def _rank_anchors(self, history):
        anchors = []
        for rank, (track, listened_time) in enumerate(history):
            if 0 <= track < len(self.neighbors):
                recency = 1.0 / (1.0 + rank)
                score = recency * max(float(listened_time), 1e-3)
                anchors.append((score, track))
        anchors.sort(reverse=True)
        return [track for _, track in anchors]

    def _load_user_history(self, user: int):
        key = f"user:{user}:listens"
        raw_entries = self.listen_history_redis.lrange(key, 0, -1)

        history = []
        for raw in raw_entries[:MAX_HISTORY]:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            entry = json.loads(raw)
            history.append((int(entry["track"]), float(entry["time"])))
        return history

    def _recommend_popular(self, seen_tracks, user, prev_track, prev_track_time):
        for track in self.popular_tracks:
            if track not in seen_tracks:
                return track
        return self.fallback_recommender.recommend_next(user, prev_track, prev_track_time)
