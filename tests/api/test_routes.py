"""测试 FastAPI 路由（使用 TestClient，不需要 Milvus）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import set_agent, _agent


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_agent():
    import src.api.dependencies as dep
    dep._agent = None
    dep._agent_pdf = ""
    yield
    dep._agent = None
    dep._agent_pdf = ""


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "no_document"

    def test_health_response_schema(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "0.2.0"
        assert data["milvus_connected"] is False
        assert data["model_loaded"] is False

    def test_metrics_endpoint(self, client):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200


class TestSearch:
    def test_search_without_agent_returns_503(self, client):
        response = client.post("/api/v1/search", json={
            "query": "什么是深度学习",
            "strategy": "hybrid",
            "top_k": 5,
        })
        assert response.status_code == 503
        assert "尚未就绪" in response.json()["detail"]

    def test_search_invalid_strategy(self, client):
        response = client.post("/api/v1/search", json={
            "query": "test",
            "strategy": "invalid",
            "top_k": 5,
        })
        assert response.status_code == 422

    def test_search_empty_query(self, client):
        response = client.post("/api/v1/search", json={
            "query": "",
            "strategy": "hybrid",
            "top_k": 5,
        })
        assert response.status_code == 422


class TestDocs:
    def test_docs_endpoint_exists(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_exists(self, client):
        response = client.get("/redoc")
        assert response.status_code == 200
