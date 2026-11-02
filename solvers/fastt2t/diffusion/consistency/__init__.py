from diffusion.consistency.mis import MISConsistency

__all__ = ["MISConsistency", "TSPConsistency"]


def __getattr__(name):
    if name == "TSPConsistency":
        from diffusion.consistency.tsp import TSPConsistency

        return TSPConsistency
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
