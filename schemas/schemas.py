from pydantic import BaseModel

class SensorDataIn(BaseModel):
    soil_moisture: float
    temperature: float
    humidity: float
    timestamp: str = None  # Optional, will default to current time if not provided