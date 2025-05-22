from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    sample_input = {
        "CreditScore": 10,
        "Geography": "Egypt",
        "Gender": "M",
        "Age": 30,
        "Tenure": 2,
        "Balance": 3000,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 1, 
        "EstimatedSalary": 5000
        }
    
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
