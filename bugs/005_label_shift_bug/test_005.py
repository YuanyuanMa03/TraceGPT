"""Test for Bug 005: Label shift bug."""

import numpy as np
import pytest


def test_wrong_labels_match_input():
    """Without shifting, labels == input (learns identity, not prediction)."""
    tokens = np.array([10, 20, 30, 40])
    x, y = tokens, tokens
    np.testing.assert_array_equal(x, y)  # This is the BUG


def test_correct_labels_are_shifted():
    """With correct shift, labels[i] = input[i+1]."""
    tokens = np.array([10, 20, 30, 40])
    x, y = tokens[:-1], tokens[1:]
    # input[0]=10, label[0]=20 → predict "love" from "I"
    assert y[0] == 20
    assert y[1] == 30
    assert y[2] == 40
    assert len(x) == len(y)


def test_no_label_is_its_own_input():
    """For next-token prediction, no label should equal its corresponding input."""
    tokens = np.array([10, 20, 30, 40])
    x, y = tokens[:-1], tokens[1:]
    for i in range(len(x)):
        assert x[i] != y[i], f"Position {i}: label == input, not shifted!"
