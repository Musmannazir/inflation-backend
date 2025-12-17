from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    """Check if the API root is accessible"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Pakistan Inflation Predictor API running"}

def test_predict_endpoint():
    """Check if the prediction endpoint works with valid data"""
    payload = {
        "t1": 12.5,
        "t2": 11.0,
        "t3": 10.5,
        "CPI_MoM": 0.5,
        "WPI_MoM": 1.2,
        "SPI_MoM": 0.8,
        "month": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_inflation" in response.json()
    assert "risk_level" in response.json()