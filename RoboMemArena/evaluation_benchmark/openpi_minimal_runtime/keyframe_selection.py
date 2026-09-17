from typing import List, Dict
from PIL import Image

def build_visual_memory(J_hist: List[List[int]], t:int, N:int, d: int) -> List[int]:
    """
    Algorithm 1: Selecting Keyframes from Candidates
    Args:
        J_hist: list over time of candidate index sets; each J_hist[t] is a list[int] (absolute frame indices)
        d: merge distance (frames). If consecutive indices differ <= d, put them into the same cluster.
    Returns:
        K_selected: list[int], selected absolute frame indices (one median per cluster)
    """
    # 1) Extract & sort temporal indices (unique)
    G = sorted(idx for J in J_hist for idx in J)
    if not G:
        return []

    # 2) Build clusters by temporal proximity (≤ d)
    clusters: List[List[int]] = []
    cur = [G[0]]
    for i in range(1, len(G)):
        if G[i] - G[i - 1] <= d:
            cur.append(G[i])
        else:
            clusters.append(cur)
            cur = [G[i]]
    clusters.append(cur)

    # 3) Select the median index of each cluster
    K_selected: List[int] = []
    for C in clusters:
        mid = len(C) // 2
        K_selected.append(C[mid])

    cutoff = t - N + 1 # first idx of recent window
    K_selected = [k for k in K_selected if k <= cutoff]

    return K_selected


def get_frames_from_indices(indices: List[int], store: Dict[int, Image.Image]) -> List[Image.Image]:
    """
    Map absolute indices -> images using your global frame store.
    Missing indices are skipped.
    """
    return [store[i] for i in indices if i in store]
