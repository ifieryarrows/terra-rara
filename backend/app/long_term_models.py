"""SQLAlchemy ORM model for persisted Long-Term Outlook snapshots."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, JSON, String

from app.db import Base


def _utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


BIGINT_PK_TYPE = BigInteger().with_variant(Integer, "sqlite")


class LongTermOutlookSnapshot(Base):
    """Persisted 20D/60D long-term copper outlook produced by the pipeline worker."""

    __tablename__ = "long_term_outlook_snapshots"

    id = Column(BIGINT_PK_TYPE, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True, default="HG=F")
    as_of_date = Column(String(10), nullable=False, index=True)
    payload_json = Column(JSON, nullable=False)
    validation_json = Column(JSON, nullable=True)
    model_version = Column(String(80), nullable=False, default="lt_structural_v1")
    generated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)
    run_id = Column(String(80), nullable=True, index=True)

    __table_args__ = (
        Index("ix_lt_outlook_symbol_generated", "symbol", "generated_at"),
        Index("ix_lt_outlook_symbol_asof", "symbol", "as_of_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<LongTermOutlookSnapshot(symbol={self.symbol}, "
            f"as_of={self.as_of_date}, version={self.model_version})>"
        )
