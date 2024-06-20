import json
import pytest
import requests
from api_server.app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/health')
    data = json.loads(response.data.decode('utf-8'))

    assert response.status_code == 200
    assert data['status'] == 'healthy'


def test_sample_api(client):
    response = client.get('/data-collection-service/api_1')
    assert response.status_code == 200
    
