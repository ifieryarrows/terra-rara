"""The deterministic commentary fallback must not invent facts.

Two production defects lived here:
- the driver lines printed `` `unknown_driver` `` placeholders when the run had
  no ranked feature drivers;
- the news line claimed "News sample size (271) may be insufficient" for a
  271-headline week, which is the opposite of what the number says.
"""

from app import commentary as commentary_module


def _fallback(
    top_influencers=None,
    news_count=271,
    sentiment_index=-0.0287,
    sentiment_label="Neutral",
):
    return commentary_module._build_commentary_template_fallback(
        current_price=6.4815,
        baseline_price=6.5430,
        predicted_price=6.5497,
        predicted_return=0.001,
        sentiment_index=sentiment_index,
        sentiment_label=sentiment_label,
        top_influencers=top_influencers if top_influencers is not None else [],
        news_count=news_count,
    )


class TestDriverLines:
    def test_no_driver_placeholder_is_ever_printed(self):
        text = _fallback(top_influencers=[])

        assert "unknown_driver" not in text

    def test_missing_driver_ranking_is_stated_plainly(self):
        text = _fallback(top_influencers=[])

        assert "driver" in text.lower()

    def test_driver_labels_are_used_when_the_run_provides_them(self):
        text = _fallback(
            top_influencers=[
                {
                    "feature": "DX_Y_NYB_ret1",
                    "label": "US Dollar Index · 1-day price change",
                    "importance": 0.31,
                },
                {
                    "feature": "GXC_lag_ret1_1",
                    "label": "GXC · return lag 1 days",
                    "importance": 0.09,
                },
            ]
        )

        assert "US Dollar Index" in text
        assert "GXC" in text

    def test_feature_key_is_used_when_no_label_is_available(self):
        text = _fallback(
            top_influencers=[{"feature": "HG=F_rsi14", "importance": 0.2}]
        )

        assert "HG=F_rsi14" in text


class TestNewsSampleLine:
    def test_a_broad_sample_does_not_claim_to_be_insufficient(self):
        text = _fallback(news_count=271)

        assert "insufficient" not in text

    def test_a_thin_sample_says_so(self):
        text = _fallback(news_count=3)

        assert "thin" in text.lower()

    def test_an_empty_sample_says_the_news_mix_is_unavailable(self):
        text = _fallback(news_count=0)

        assert "no headlines" in text.lower()
