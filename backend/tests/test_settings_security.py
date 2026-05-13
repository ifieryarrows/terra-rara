from app.settings import Settings


def test_cors_origins_default_is_not_wildcard():
    settings = Settings()
    assert "*" not in settings.cors_allowed_origins_list


def test_cors_origins_parse_env_style_csv():
    settings = Settings(cors_allowed_origins="https://app.example.com, https://admin.example.com")
    assert settings.cors_allowed_origins_list == [
        "https://app.example.com",
        "https://admin.example.com",
    ]
