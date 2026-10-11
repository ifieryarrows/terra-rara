"""Tests for database URL normalization and driver compatibility."""

from unittest.mock import patch, MagicMock
import pytest

from app.db import normalize_database_url, get_db_type


def test_sqlite_url_normalization():
    url, connect_args = normalize_database_url("sqlite:////data/app.db")
    assert url == "sqlite:////data/app.db"
    assert connect_args == {}

    url, connect_args = normalize_database_url("sqlite:///:memory:")
    assert url == "sqlite:///:memory:"
    assert connect_args == {}


def test_postgres_legacy_scheme_replacement():
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.side_effect = lambda name: MagicMock() if name == "psycopg2" else None
        url, connect_args = normalize_database_url("postgres://user:pass@localhost:5432/testdb")
        assert url == "postgresql+psycopg2://user:pass@localhost:5432/testdb"
        assert connect_args == {}


def test_psycopg_available_sets_psycopg_driver_and_pooler_args():
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.side_effect = lambda name: MagicMock() if name in ("psycopg", "psycopg2") else None

        # Standard non-pooler connection
        url, connect_args = normalize_database_url("postgresql://user:pass@localhost:5432/testdb")
        assert url == "postgresql+psycopg://user:pass@localhost:5432/testdb"
        assert connect_args == {}

        # Supabase pooler on port 6543
        pooler_url = "postgresql://postgres.test:pass@aws-1-eu-central-2.pooler.supabase.com:6543/postgres"
        url, connect_args = normalize_database_url(pooler_url)
        assert url.startswith("postgresql+psycopg://")
        assert connect_args.get("prepare_threshold") is None


def test_psycopg2_fallback_when_psycopg_not_installed():
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.side_effect = lambda name: MagicMock() if name == "psycopg2" else None

        # Plain postgresql:// URL falls back to psycopg2
        url, connect_args = normalize_database_url("postgresql://user:pass@localhost:5432/testdb")
        assert url == "postgresql+psycopg2://user:pass@localhost:5432/testdb"
        assert connect_args == {}

        # postgresql+psycopg:// explicitly requested falls back to psycopg2
        url, connect_args = normalize_database_url("postgresql+psycopg://user:pass@localhost:5432/testdb")
        assert url == "postgresql+psycopg2://user:pass@localhost:5432/testdb"
        assert connect_args == {}


def test_psycopg_fallback_when_psycopg2_not_installed():
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.side_effect = lambda name: MagicMock() if name == "psycopg" else None

        # Plain postgresql:// URL uses psycopg
        url, connect_args = normalize_database_url("postgresql://user:pass@localhost:5432/testdb")
        assert url == "postgresql+psycopg://user:pass@localhost:5432/testdb"

        # postgresql+psycopg2:// explicitly requested falls back to psycopg
        url, connect_args = normalize_database_url("postgresql+psycopg2://user:pass@localhost:5432/testdb")
        assert url == "postgresql+psycopg://user:pass@localhost:5432/testdb"


def test_explicit_driver_preserved_when_available():
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.side_effect = lambda name: MagicMock() if name in ("psycopg", "psycopg2") else None

        # Explicit psycopg2 is preserved
        url, connect_args = normalize_database_url("postgresql+psycopg2://user:pass@localhost:5432/testdb")
        assert url == "postgresql+psycopg2://user:pass@localhost:5432/testdb"
        assert connect_args == {}

        # Explicit psycopg is preserved
        url, connect_args = normalize_database_url("postgresql+psycopg://user:pass@localhost:5432/testdb")
        assert url == "postgresql+psycopg://user:pass@localhost:5432/testdb"


def test_get_db_type_resolution():
    with patch("app.db.get_settings") as mock_settings:
        mock_settings.return_value.database_url = "sqlite:////data/app.db"
        assert get_db_type() == "sqlite"

        mock_settings.return_value.database_url = "postgresql://user:pass@localhost:5432/db"
        assert get_db_type() == "postgresql"

        mock_settings.return_value.database_url = "postgresql+psycopg://user:pass@localhost:5432/db"
        assert get_db_type() == "postgresql"

        mock_settings.return_value.database_url = "postgres://user:pass@localhost:5432/db"
        assert get_db_type() == "postgresql"

        mock_settings.return_value.database_url = "mysql://user:pass@localhost:3306/db"
        assert get_db_type() == "mysql"

        mock_settings.return_value.database_url = "oracle://user:pass@localhost/db"
        assert get_db_type() == "unknown"


def test_get_engine_creates_engine_with_normalized_driver():
    import app.db
    app.db._engine = None  # Reset singleton

    with patch("app.db.get_settings") as mock_settings:
        mock_settings.return_value.database_url = "sqlite:///:memory:"
        mock_settings.return_value.log_level = "INFO"
        engine = app.db.get_engine()
        assert engine.url.drivername == "sqlite"
        assert app.db.check_db_connection() is True

    app.db._engine = None  # Reset singleton
    with patch("app.db.get_settings") as mock_settings, \
         patch("importlib.util.find_spec") as mock_find_spec, \
         patch("app.db.create_engine") as mock_create_engine:
        mock_find_spec.side_effect = lambda name: MagicMock() if name == "psycopg2" else None
        mock_settings.return_value.database_url = "postgresql://user:pass@aws-1-eu-central-2.pooler.supabase.com:6543/postgres"
        mock_settings.return_value.log_level = "INFO"

        app.db.get_engine()
        mock_create_engine.assert_called_once()
        call_url, call_kwargs = mock_create_engine.call_args[0][0], mock_create_engine.call_args[1]
        assert call_url.startswith("postgresql+psycopg2://")
        assert call_kwargs["pool_size"] == 5

    app.db._engine = None  # Reset singleton
    with patch("app.db.get_settings") as mock_settings, \
         patch("importlib.util.find_spec") as mock_find_spec, \
         patch("app.db.create_engine") as mock_create_engine:
        mock_find_spec.side_effect = lambda name: MagicMock() if name in ("psycopg", "psycopg2") else None
        mock_settings.return_value.database_url = "postgresql://user:pass@aws-1-eu-central-2.pooler.supabase.com:6543/postgres"
        mock_settings.return_value.log_level = "INFO"

        app.db.get_engine()
        mock_create_engine.assert_called_once()
        call_url, call_kwargs = mock_create_engine.call_args[0][0], mock_create_engine.call_args[1]
        assert call_url.startswith("postgresql+psycopg://")
        assert call_kwargs["connect_args"]["prepare_threshold"] is None


def test_empty_and_none_database_url_normalization():
    assert normalize_database_url(None) == ("", {})
    assert normalize_database_url("") == ("", {})


def test_neither_driver_installed_behavior():
    with patch("importlib.util.find_spec", return_value=None):
        url, connect_args = normalize_database_url("postgresql://user:pass@localhost:5432/testdb")
        assert url == "postgresql://user:pass@localhost:5432/testdb"
        assert connect_args == {}

        # If explicit psycopg requested but neither installed, prepare_threshold is not added
        url, connect_args = normalize_database_url("postgresql+psycopg://user:pass@localhost:6543/testdb")
        assert url == "postgresql+psycopg://user:pass@localhost:6543/testdb"
        assert connect_args == {}


def test_supabase_domain_triggers_prepare_threshold_for_psycopg():
    with patch("importlib.util.find_spec") as mock_find_spec:
        mock_find_spec.side_effect = lambda name: MagicMock() if name == "psycopg" else None
        supabase_url = "postgresql://postgres:pass@db.xyz.supabase.co:5432/postgres"
        url, connect_args = normalize_database_url(supabase_url)
        assert url.startswith("postgresql+psycopg://")
        assert connect_args.get("prepare_threshold") is None


def test_get_db_type_handles_none_and_empty():
    with patch("app.db.get_settings") as mock_settings:
        mock_settings.return_value.database_url = None
        assert get_db_type() == "unknown"

        mock_settings.return_value.database_url = ""
        assert get_db_type() == "unknown"

