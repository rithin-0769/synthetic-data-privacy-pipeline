"""API integration tests for FastAPI synthetic service."""
import pytest
from fastapi.testclient import TestClient
from src.service.api import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Synthetic Data Privacy Pipeline API"

def test_profile_endpoint(client):
    response = client.get("/profile")
    assert response.status_code == 200
    data = response.json()
    assert "columns" in data
    assert "CustomerID" in data["sensitive"]

def test_generate_endpoint(client):
    response = client.post("/generate?samples=10&model_type=copula&epsilon=1.5")
    assert response.status_code == 200
    body = response.json()
    assert body["sample_count"] == 10
    assert len(body["records"]) == 10
    assert "CustomerID" in body["dropped_direct_identifiers"]
    # Check that CustomerID is not present in output records
    assert "CustomerID" not in body["records"][0]
