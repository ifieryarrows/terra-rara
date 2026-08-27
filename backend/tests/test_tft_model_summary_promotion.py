"""Regression coverage for active-model filtering on the Models API."""

import asyncio

import pytest
from fastapi import HTTPException

from app import main


def test_tft_summary_does_not_fall_back_to_rejected_candidate(monkeypatch):
    class EmptyQuery:
        def filter(self, *_args):
            return self

        def order_by(self, *_args):
            return self

        def first(self):
            return None

    class FakeSession:
        query_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def query(self, _model):
            self.query_count += 1
            return EmptyQuery()

    session = FakeSession()
    monkeypatch.setattr(main, "SessionLocal", lambda: session)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.get_tft_summary("HG=F"))

    assert exc_info.value.status_code == 404
    assert "quality-gate-passed" in str(exc_info.value.detail)
    assert session.query_count == 1
