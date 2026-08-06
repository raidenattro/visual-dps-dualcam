import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.frame_sampling import sample_frame_indices


def test_25_to_15_keeps_three_of_every_five():
    kept = sample_frame_indices(range(1, 2501), source_fps=25, target_fps=15)
    assert len(kept) == 1500
    assert kept[:6] == [1, 3, 5, 6, 8, 10]  # 间隔 +2,+2,+1,+2,+2 循环


def test_no_sampling_when_target_not_lower():
    idx = list(range(1, 11))
    assert sample_frame_indices(idx, source_fps=25, target_fps=25) == idx
    assert sample_frame_indices(idx, source_fps=25, target_fps=30) == idx
    assert sample_frame_indices(idx, source_fps=25, target_fps=0) == idx


def test_sorts_and_preserves_original_frame_numbers():
    kept = sample_frame_indices([9, 3, 7, 1, 5], source_fps=25, target_fps=15)
    assert kept == sorted(kept)
    assert set(kept) <= {1, 3, 5, 7, 9}


def test_empty_input():
    assert sample_frame_indices([], source_fps=25, target_fps=15) == []
