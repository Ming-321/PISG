import numpy as np
from pathlib import Path


def parse_tour(tokens: list[str], num_nodes: int) -> np.ndarray:
    tour = np.array(list(map(int, tokens)), dtype=np.int32)
    if tour.size == num_nodes + 1 and tour[0] == tour[-1]:
        tour = tour[:-1]

    if tour.size != num_nodes:
        raise ValueError(f"Expected {num_nodes} tour nodes, got {tour.size}")

    if tour.min() == 1 and tour.max() == num_nodes:
        tour = tour - 1
    elif tour.min() != 0 or tour.max() != num_nodes - 1:
        raise ValueError(
            f"Unexpected tour indexing range [{tour.min()}, {tour.max()}] for {num_nodes} nodes"
        )

    return tour


def compute_cost(coords: np.ndarray, tour: np.ndarray) -> np.float32:
    ordered = coords[tour]
    rolled = np.roll(ordered, shift=1, axis=0)
    edge_lengths = np.linalg.norm(ordered - rolled, axis=-1)
    return np.float32(edge_lengths.sum())


def parse_line(line: str) -> tuple[np.ndarray, np.ndarray, np.float32]:
    tokens = line.strip().split()
    out_idx = tokens.index("output")
    coords = np.array(list(map(float, tokens[:out_idx])), dtype=np.float32).reshape(-1, 2)
    tour = parse_tour(tokens[out_idx + 1 :], num_nodes=coords.shape[0])
    cost = compute_cost(coords, tour)
    return coords, tour, cost


def convert(src_path: str, save_path: str):
    coords_list: list[np.ndarray] = []
    tours_list: list[np.ndarray] = []
    costs_list: list[np.float32] = []

    with Path(src_path).open("r") as f:
        for line in f:
            if not line.strip():
                continue
            coords, tour, cost = parse_line(line)
            coords_list.append(coords)
            tours_list.append(tour)
            costs_list.append(cost)

    coords = np.stack(coords_list, axis=0)
    opt_tours = np.stack(tours_list, axis=0)
    opt_costs = np.array(costs_list, dtype=np.float32)
    np.savez_compressed(save_path, coords=coords, opt_tours=opt_tours, opt_costs=opt_costs)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    if not args.overwrite:
        assert not os.path.exists(args.save_path)

    convert(args.src_path, args.save_path)
