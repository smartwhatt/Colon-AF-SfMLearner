import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_splits import (
    dev_stratification_key,
    sample_count,
    sequence_summary,
    split_dev_sequences,
)


def _seq(name, frames=6):
    return (f"c1/ascending/phantom/{name}", frames)


def test_split_dev_sequences_is_deterministic_and_disjoint():
    train = [
        _seq("c1_ascending_t1_v1"),
        _seq("c1_ascending_t1_v2"),
        _seq("c1_cecum_t1_v1"),
        _seq("c1_cecum_t1_v2"),
        _seq("c1_ascending_t2_v1"),
        _seq("c1_ascending_t2_v2"),
        _seq("c1_cecum_t2_v1"),
        _seq("c1_cecum_t2_v2"),
    ]

    kept_a, dev_a = split_dev_sequences(train, 0.25, seed=7)
    kept_b, dev_b = split_dev_sequences(train, 0.25, seed=7)

    assert kept_a == kept_b
    assert dev_a == dev_b
    assert len(dev_a) == 2
    assert {s for s, _ in kept_a}.isdisjoint({s for s, _ in dev_a})
    assert {dev_stratification_key(s) for s, _ in dev_a} & {
        dev_stratification_key(s) for s, _ in kept_a
    } == set()


def test_sequence_summary_counts_samples_by_metadata():
    seqs = [
        _seq("c1_ascending_t1_v1", frames=6),
        _seq("c1_cecum_t2_v3", frames=5),
    ]

    summary = sequence_summary(seqs)

    assert sample_count(seqs) == 7
    assert summary["samples"] == 7
    assert summary["by_trial"] == {"t1": 4, "t2": 3}
    assert summary["by_colon"] == {"c1": 7}
    assert summary["by_variant"] == {"v1": 4, "v3": 3}
