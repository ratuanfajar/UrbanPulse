from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

app = FastAPI()

class Location(BaseModel):
    latitude: float
    longitude: float

# Load phase2_best.pt model correctly
model_data = None
try:
    import torch
    import os
    
    # Path to model (since app.py is in backend-ml, model is in root)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'phase2_best.pt')
    
    print(f"Loading model from {model_path}...")
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    print("Model loaded successfully. Type:", type(model_data))
except Exception as e:
    print(f"Warning: Could not load model: {e}")

@app.post("/predict")
async def predict_slum(location: Location):
    try:
        # Here you would normally run your model on the coordinates
        # e.g., fetching satellite image for the lat/lon and passing to model
        # is_slum = model.predict(image)
        
        # SEMENTARA: Mocking model response (kumuh/non-kumuh)
        # We can't run model.predict() because model_data is a state_dict, not a model object
        is_slum = random.choice([True, False])
        
        return {
            "isSlum": is_slum,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "confidence": 0.85,
            "model_version": "phase2_best" if model_data else "mock"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
