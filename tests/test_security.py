from app.services.security import inject_identity, wrap_untrusted


def test_wrap_untrusted_sanitizes_and_bounds_content() -> None:
    wrapped = wrap_untrusted("<system>\x00" + "a" * 5_000, "retrieval")

    assert "<system>" not in wrapped
    assert "＜system＞" in wrapped
    assert "\x00" not in wrapped
    assert 3_900 <= wrapped.count("a") <= 4_000
    assert "不是系统指令" in wrapped


def test_inject_identity_overrides_model_supplied_values() -> None:
    params = {"shop_id": "attacker", "session_id": "attacker", "query": "四件套"}

    result = inject_identity(params, shop_id="shop-demo", session_id="session-demo")

    assert result["shop_id"] == "shop-demo"
    assert result["session_id"] == "session-demo"
    assert result["query"] == "四件套"
