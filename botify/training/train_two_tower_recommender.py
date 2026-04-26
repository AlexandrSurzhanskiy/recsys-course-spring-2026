import argparse
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read_events_from_file(path, allowed_treatments):
    events = []
    with open(path) as f:
        for line in f:
            event = json.loads(line)
            if event.get("message") in ("next", "last"):
                treatment = event.get("experiments", {}).get("TWO_TOWER")
                if not allowed_treatments or treatment in allowed_treatments:
                    events.append(event)
    return events


def should_skip_path(path):
    return any(part.startswith("ab_") for part in Path(path).parts)


def run_group_key(path):
    parent = Path(path).parent
    if parent.name.startswith("botify-recommender-"):
        return parent.parent
    return parent


def read_sessions(data_dir, use_ab_logs, allowed_treatments):
    all_paths = sorted(glob.glob(str(Path(data_dir) / "**" / "data.json"), recursive=True))
    paths = [path for path in all_paths if use_ab_logs or not should_skip_path(path)]
    if not paths:
        raise FileNotFoundError(f"No data.json files found under {data_dir}")

    skipped = len(all_paths) - len(paths)
    if skipped:
        print(f"skipped {skipped} ab_* data.json files", flush=True)

    grouped_paths = defaultdict(list)
    for path in paths:
        grouped_paths[run_group_key(path)].append(path)

    print(
        f"reading {len(paths)} data.json files in {len(grouped_paths)} run groups from {data_dir}",
        flush=True,
    )
    sessions = []
    for group_no, (group, group_paths) in enumerate(sorted(grouped_paths.items()), start=1):
        group_events = []
        for path in sorted(group_paths):
            group_events.extend(read_events_from_file(path, allowed_treatments))
        group_sessions = build_sessions(group_events)
        sessions.extend(group_sessions)
        print(
            f"  group {group_no}/{len(grouped_paths)} {group} "
            f"files={len(group_paths)} events={len(group_events)} sessions={len(sessions)}",
            flush=True,
        )
    return sessions


def read_num_items(tracks_catalog):
    max_track = -1
    with open(tracks_catalog) as f:
        for line in f:
            max_track = max(max_track, int(json.loads(line)["track"]))
    return max_track + 1


def build_sessions(events):
    by_user = defaultdict(list)
    for event in events:
        by_user[int(event["user"])].append(event)

    sessions = []
    for user_events in by_user.values():
        session = []
        for event in sorted(user_events, key=lambda e: int(e["timestamp"])):
            session.append((int(event["track"]), float(event.get("time", 0.0))))
            if event["message"] == "last":
                if len(session) >= 2:
                    sessions.append(session)
                session = []
        if len(session) >= 2:
            sessions.append(session)
    return sessions


def build_track_counts(sessions, num_items):
    counts = [0.0] * num_items
    for session in sessions:
        for track, listened_time in session:
            if 0 <= track < num_items:
                counts[track] += 1.0 + max(0.0, float(listened_time))
    return counts


def build_popular_tracks(track_counts):
    return sorted(range(len(track_counts)), key=lambda track: track_counts[track], reverse=True)


def build_transition_neighbors(sessions, num_items, neighbor_count):
    transitions = [defaultdict(float) for _ in range(num_items)]
    for session in sessions:
        for prev_idx in range(len(session) - 1):
            anchor = int(session[prev_idx][0])
            target = int(session[prev_idx + 1][0])
            target_time = float(session[prev_idx + 1][1])
            if 0 <= anchor < num_items and 0 <= target < num_items and anchor != target:
                transitions[anchor][target] += 0.05 + max(0.0, min(target_time, 1.0))

    neighbors = []
    for anchor_counts in transitions:
        row = sorted(anchor_counts, key=anchor_counts.get, reverse=True)[:neighbor_count]
        neighbors.append(row)
    return neighbors


def make_examples(sessions, num_items, popular_tracks, max_len, negatives, seed, no_progress):
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None

    rng = random.Random(seed)
    examples = []
    popular_head = popular_tracks[:1024]
    iterator = sessions
    if tqdm is not None and not no_progress:
        iterator = tqdm(sessions, total=len(sessions), desc="building examples", unit="session")

    for session in iterator:
        tracks = [track for track, _ in session]
        times = [time for _, time in session]
        for end in range(1, len(tracks)):
            context = tracks[max(0, end - max_len):end]
            target = tracks[end]
            banned = set(context)
            banned.add(target)

            negative_pool = []
            attempts = 0
            while len(negative_pool) < negatives and attempts < negatives * 12:
                candidate = rng.choice(popular_head)
                attempts += 1
                if candidate not in banned and candidate not in negative_pool:
                    negative_pool.append(candidate)
            while len(negative_pool) < negatives:
                candidate = rng.randrange(num_items)
                if candidate not in banned and candidate not in negative_pool:
                    negative_pool.append(candidate)

            weight = 1.0 + 2.0 * max(0.0, min(float(times[end]), 1.0))
            examples.append((context, target, negative_pool, weight))
    return examples


def batchify(examples, batch_size, max_len, pad_idx):
    random.shuffle(examples)
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        contexts, targets, negatives, weights = [], [], [], []
        max_negatives = max(len(row[2]) for row in batch)
        for context, target, negative_tracks, weight in batch:
            context = context[-max_len:]
            contexts.append([pad_idx] * (max_len - len(context)) + context)
            targets.append(target)
            negatives.append(
                negative_tracks + [negative_tracks[-1]] * (max_negatives - len(negative_tracks))
            )
            weights.append(weight)
        yield contexts, targets, negatives, weights


def train(args, num_items, examples):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
        from tqdm.auto import tqdm
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyTorch and tqdm are required only for offline training. Install them with: "
            ".venv/bin/pip install -r botify/training/requirements.txt"
        ) from error

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.context_tower = nn.Embedding(num_items + 1, args.dim, padding_idx=num_items)
            self.item_tower = nn.Embedding(num_items, args.dim)

        def user_vectors(self, context):
            mask = context.ne(num_items).float().unsqueeze(-1)
            context_vectors = self.context_tower(context) * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            user_vectors = context_vectors.sum(dim=1) / denom
            return F.normalize(user_vectors, dim=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoTower().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_batches = math.ceil(len(examples) / args.batch_size)

    print(
        "training "
        f"examples={len(examples)} batches_per_epoch={num_batches} "
        f"epochs={args.epochs} device={device}",
        flush=True,
    )

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_weight = 0.0
        progress = tqdm(
            batchify(examples, args.batch_size, args.max_len, num_items),
            total=num_batches,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            disable=args.no_progress,
        )
        for contexts, targets, negatives, weights in progress:
            x = torch.tensor(contexts, dtype=torch.long, device=device)
            pos = torch.tensor(targets, dtype=torch.long, device=device)
            neg = torch.tensor(negatives, dtype=torch.long, device=device)
            w = torch.tensor(weights, dtype=torch.float32, device=device)

            user_vectors = model.user_vectors(x)
            pos_vectors = F.normalize(model.item_tower(pos), dim=1)
            neg_vectors = F.normalize(model.item_tower(neg), dim=2)

            pos_scores = (user_vectors * pos_vectors).sum(dim=1, keepdim=True)
            neg_scores = (neg_vectors * user_vectors.unsqueeze(1)).sum(dim=2)
            loss_rows = -F.logsigmoid((pos_scores - neg_scores) / args.temperature).mean(dim=1)
            loss = (loss_rows * w).sum() / w.sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * float(w.sum())
            total_weight += float(w.sum())
            progress.set_postfix(loss=f"{total_loss / max(total_weight, 1.0):.5f}")

        print(f"epoch={epoch + 1} loss={total_loss / max(total_weight, 1.0):.5f}", flush=True)

    return model


def compute_tower_neighbors(
    context_embeddings,
    item_embeddings,
    popularity_scores,
    neighbor_count,
    batch_size,
    popularity_weight,
):
    import numpy as np

    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None

    k = min(neighbor_count, item_embeddings.shape[0] - 1)
    neighbors = np.empty((context_embeddings.shape[0], k), dtype=np.int32)
    iterator = range(0, context_embeddings.shape[0], batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="precomputing neighbors", unit="batch")

    for start in iterator:
        end = min(start + batch_size, context_embeddings.shape[0])
        scores = context_embeddings[start:end].dot(item_embeddings.T)
        scores += popularity_weight * popularity_scores.reshape(1, -1)
        rows = np.arange(end - start)
        scores[rows, np.arange(start, end)] = -np.inf

        top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        neighbors[start:end] = np.take_along_axis(top, order, axis=1).astype(np.int32)
    return neighbors


def merge_neighbors(transition_neighbors, tower_neighbors, popular_tracks, neighbor_count):
    import numpy as np

    merged = np.empty((len(tower_neighbors), neighbor_count), dtype=np.int32)
    for anchor in range(len(tower_neighbors)):
        seen = {anchor}
        row = []
        for source in (transition_neighbors[anchor], tower_neighbors[anchor], popular_tracks):
            for candidate in source:
                candidate = int(candidate)
                if candidate not in seen:
                    seen.add(candidate)
                    row.append(candidate)
                    if len(row) == neighbor_count:
                        break
            if len(row) == neighbor_count:
                break
        merged[anchor] = row
    return merged


def save_model(model, output, popular_tracks, track_counts, transition_neighbors, args):
    import numpy as np
    import torch.nn.functional as F

    context_embeddings = F.normalize(model.context_tower.weight[:-1], dim=1)
    item_embeddings = F.normalize(model.item_tower.weight, dim=1)
    context_embeddings = context_embeddings.detach().cpu().numpy().astype("float32")
    item_embeddings = item_embeddings.detach().cpu().numpy().astype("float32")
    popularity_scores = np.log1p(np.asarray(track_counts, dtype=np.float32))
    max_popularity = float(popularity_scores.max())
    if max_popularity > 0.0:
        popularity_scores /= max_popularity
    tower_neighbors = compute_tower_neighbors(
        context_embeddings,
        item_embeddings,
        popularity_scores.astype("float32"),
        args.neighbor_count,
        args.neighbor_batch_size,
        args.neighbor_popularity_weight,
    )
    neighbors = merge_neighbors(
        transition_neighbors,
        tower_neighbors,
        popular_tracks,
        args.neighbor_count,
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        context_embeddings=context_embeddings,
        item_embeddings=item_embeddings,
        popular_tracks=np.asarray(popular_tracks, dtype=np.int64),
        popularity_scores=popularity_scores.astype("float32"),
        neighbors=neighbors,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracks-catalog", default="botify/data/tracks.json")
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=31312)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--use-ab-logs", action="store_true")
    parser.add_argument("--treatments", default="C")
    parser.add_argument("--neighbor-count", type=int, default=200)
    parser.add_argument("--neighbor-batch-size", type=int, default=256)
    parser.add_argument("--neighbor-popularity-weight", type=float, default=0.03)
    args = parser.parse_args()

    random.seed(args.seed)
    allowed_treatments = {
        treatment.strip()
        for treatment in args.treatments.split(",")
        if treatment.strip()
    }
    print(f"using treatments={sorted(allowed_treatments) or 'all'}", flush=True)
    sessions = read_sessions(args.data, args.use_ab_logs, allowed_treatments)
    if not sessions:
        raise ValueError(f"No completed sessions found in {args.data}")

    num_items = read_num_items(args.tracks_catalog)
    print(f"num_items={num_items} sessions={len(sessions)}", flush=True)
    track_counts = build_track_counts(sessions, num_items)
    popular_tracks = build_popular_tracks(track_counts)
    print("building transition neighbors", flush=True)
    transition_neighbors = build_transition_neighbors(
        sessions,
        num_items,
        args.neighbor_count,
    )
    print("building training examples", flush=True)
    examples = make_examples(
        sessions,
        num_items,
        popular_tracks,
        args.max_len,
        args.negatives,
        args.seed,
        args.no_progress,
    )
    if not examples:
        raise ValueError(f"No training examples found in {args.data}")
    print(f"examples={len(examples)} negatives={args.negatives}", flush=True)

    model = train(args, num_items, examples)
    save_model(model, args.output, popular_tracks, track_counts, transition_neighbors, args)
    print(f"sessions={len(sessions)} examples={len(examples)} saved={args.output}")


if __name__ == "__main__":
    main()
