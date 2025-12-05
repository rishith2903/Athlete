import os
import io
from fastapi.testclient import TestClient
from backend.models.api_services.pose_service import app

def test_analyze_returns_schema():
    client = TestClient(app)
    # Create a dummy image in memory
    from PIL import Image
    img = Image.new('RGB', (64, 64), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)

    files = { 'file': ('test.jpg', buf, 'image/jpeg') }
    resp = client.post('/analyze', files=files, data={'exercise_type': 'squat'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'formScore' in data
    assert 'feedback' in data
    assert 'corrections' in data
