"""Tests for FinBERT embedding utilities (offline, no model loading)."""

import numpy as np
import pytest

from deep_learning.data.embeddings import (
    embedding_to_bytes,
    bytes_to_embedding,
    aggregate_daily_embeddings,
)


def test_embedding_roundtrip():
    vec = np.random.randn(768).astype(np.float32)
    data = embedding_to_bytes(vec)
    recovered = bytes_to_embedding(data, dim=768)
    np.testing.assert_array_almost_equal(vec, recovered)


def test_embedding_to_bytes_size():
    vec = np.random.randn(32).astype(np.float32)
    data = embedding_to_bytes(vec)
    assert len(data) == 32 * 4


def test_aggregate_single_embedding():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = aggregate_daily_embeddings(vec)
    np.testing.assert_array_equal(vec, result)


def test_aggregate_multiple_embeddings_mean():
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = aggregate_daily_embeddings(vecs)
    np.testing.assert_array_almost_equal(result, [2.0, 3.0])


def test_aggregate_weighted():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    weights = np.array([0.75, 0.25])
    result = aggregate_daily_embeddings(vecs, weights=weights)
    np.testing.assert_array_almost_equal(result, [0.75, 0.25])
