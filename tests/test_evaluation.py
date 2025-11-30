import torch

from src.evaluation.evaluate import greedy_decode


def test_greedy_decode_collapses_repeats_and_blanks() -> None:
    # Batch of 2, time 5, vocab 3 (blank=0)
    # Sequence 1: tokens 1 1 2 2 2 -> collapse to [1, 2]
    # Sequence 2: tokens 0(blank) 1 1 0 2 -> collapse to [1, 2]
    log_probs = torch.nn.functional.one_hot(
        torch.tensor([[1, 1, 2, 2, 2], [0, 1, 1, 0, 2]]), num_classes=3
    ).float()
    decoded = greedy_decode(log_probs, torch.tensor([5, 5]), blank_id=0)
    assert decoded == [[1, 2], [1, 2]]
