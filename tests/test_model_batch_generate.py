from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dflash.model import dflash_generate


class DummyDraft:
    block_size = 1
    mask_token_id = 0


class DummyTarget(nn.Module):
    def __init__(self, *, vocab_size: int = 16, stop_token_id: int = 9):
        super().__init__()
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.stop_token_id = stop_token_id
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        if kwargs.get("logits_to_keep") == 1:
            seq_len = 1
        logits = torch.full((batch_size, seq_len, self.vocab_size), -1000.0)
        for row in range(batch_size):
            if self.calls == 0:
                token = 1 + row
            elif row == 1 and self.calls == 2:
                token = self.stop_token_id
            else:
                token = 2 + self.calls + row
            logits[row, :, token % self.vocab_size] = 1000.0
        self.calls += 1
        return SimpleNamespace(logits=logits)


def test_num_return_sequences_expands_single_prompt():
    target = DummyTarget()
    output = dflash_generate(
        DummyDraft(),
        target=target,
        input_ids=torch.tensor([[4, 5]]),
        max_new_tokens=3,
        stop_token_ids=None,
        temperature=0.0,
        num_return_sequences=4,
    )

    assert output.shape == (4, 5)
    assert output[:, :2].tolist() == [[4, 5], [4, 5], [4, 5], [4, 5]]
    assert len({tuple(row.tolist()) for row in output[:, 2:]}) > 1


def test_num_return_sequences_records_per_row_completion_lengths(monkeypatch):
    monkeypatch.setattr("dflash.model._cuda_time", lambda: 0.0)
    target = DummyTarget(stop_token_id=9)
    output = dflash_generate(
        DummyDraft(),
        target=target,
        input_ids=torch.tensor([[4, 5]]),
        max_new_tokens=4,
        stop_token_ids=[9],
        temperature=0.0,
        num_return_sequences=3,
        return_stats=True,
    )

    assert output.output_ids.shape == (3, 6)
    assert output.completion_lengths == [4, 3, 4]


def test_num_return_sequences_masks_tokens_after_row_stop():
    target = DummyTarget(stop_token_id=9)
    output = dflash_generate(
        DummyDraft(),
        target=target,
        input_ids=torch.tensor([[4, 5]]),
        max_new_tokens=4,
        stop_token_ids=[9],
        temperature=0.0,
        num_return_sequences=3,
    )

    assert output.shape == (3, 6)
    assert output[1].tolist() == [4, 5, 2, 4, 9, 0]


def test_multi_prompt_multi_sample_records_prompt_lengths(monkeypatch):
    monkeypatch.setattr("dflash.model._cuda_time", lambda: 0.0)
    output = dflash_generate(
        DummyDraft(),
        target=DummyTarget(),
        input_ids=torch.tensor([[0, 4, 5], [6, 7, 8]]),
        attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        max_new_tokens=2,
        stop_token_ids=None,
        temperature=0.0,
        num_return_sequences=2,
        return_stats=True,
    )

    assert output.output_ids.shape == (4, 5)
    assert output.prompt_lengths == [2, 2, 3, 3]
    assert output.output_ids[:, :3].tolist() == [
        [0, 4, 5],
        [0, 4, 5],
        [6, 7, 8],
        [6, 7, 8],
    ]


def test_multi_prompt_requires_left_padding():
    with pytest.raises(ValueError, match="left-padded"):
        dflash_generate(
            DummyDraft(),
            target=DummyTarget(),
            input_ids=torch.tensor([[4, 5, 0], [6, 7, 8]]),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
            max_new_tokens=1,
            stop_token_ids=None,
            temperature=0.0,
            num_return_sequences=1,
        )


def test_attention_mask_shape_must_match_input_ids():
    with pytest.raises(ValueError, match="same shape"):
        dflash_generate(
            DummyDraft(),
            target=DummyTarget(),
            input_ids=torch.tensor([[1], [2]]),
            attention_mask=torch.tensor([[1]]),
            max_new_tokens=1,
            stop_token_ids=None,
            temperature=0.0,
            num_return_sequences=2,
        )
