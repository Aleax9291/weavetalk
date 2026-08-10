from app.agent.tools import analyze_reviews, check_inventory, query_order


def test_query_order_uses_synthetic_fixture() -> None:
    result = query_order("DD20260801001")

    assert result["ok"] is True
    assert result["order"]["order_id"] == "DD20260801001"


def test_query_order_rejects_unknown_order() -> None:
    result = query_order("DD00000000000")

    assert result["ok"] is False


def test_review_metrics_are_grounded_and_bounded() -> None:
    result = analyze_reviews(["P1001"])
    summary = result["summary"]["P1001"]

    assert result["ok"] is True
    assert summary["total"] > 0
    assert 0 <= summary["positive_rate"] <= 1


def test_inventory_returns_all_skus_for_known_product() -> None:
    result = check_inventory("P1001")

    assert result["ok"] is True
    assert result["skus"]
    assert all({"sku_id", "color", "stock", "price"} <= sku.keys() for sku in result["skus"])
