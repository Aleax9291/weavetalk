import re

from fastapi.testclient import TestClient

from app.api.chat import SESSIONS
from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_session_returns_server_generated_identifier() -> None:
    response = client.post("/api/session")
    session_id = response.json()["session_id"]

    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{32}", session_id)
    assert session_id in SESSIONS


def test_unknown_session_is_rejected() -> None:
    response = client.post(
        "/api/chat",
        json={"message": "你好", "session_id": "0" * 32},
    )

    assert response.status_code == 404


def test_oversized_message_is_rejected_before_model_call() -> None:
    response = client.post("/api/chat", json={"message": "a" * 2_001})

    assert response.status_code == 422


def test_mock_order_endpoint_returns_only_fixture_data() -> None:
    response = client.get("/api/mock/orders/DD20260801001")

    assert response.status_code == 200
    assert response.json()["data"]["order_id"] == "DD20260801001"
