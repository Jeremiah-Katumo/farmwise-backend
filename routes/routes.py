from fastapi import APIRouter, Depends, Body, File, UploadFile
from ..schemas import schemas
from ..cruds import cruds
from ..database import get_db
from sqlalchemy.orm import Session
import pandas as pd
from joblib import load
import os, pathlib, json, openai
from dotenv import load_dotenv
from ..utils import utils
from datetime import datetime


model = load("model.pkl")

router = APIRouter()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@router.post("/sensor-data/")
async def receive_sensor_data(data: schemas.SensorDataIn, db: Session = Depends(get_db)):
    return cruds.save_data(db, data)


@router.post("/predict/")
async def predict_yield(data: schemas.SensorDataIn):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"predicted_yield": prediction[0]}


@router.post("/chat/")
async def chatbot(data: dict = Body(...)):
    question = data.get("question")
    category = data.get("category", "crop")
    prompt = utils.category_prompts.get(category, "You are a helpful assistant.")
    prompt += f"\nUser: {question}\nAssistant:"
    
    if not openai.api_key:
        return {"answer": utils.offline_faq.get(category, {}).get(question.lower(), "No offline response.")}
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=150,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        # answer = response["choices"][0]["message"]["content"]
        
        # Cache the answer
        timestamp = datetime.utcnow().isoformat()
        pathlib.Path(f"{CACHE_DIR}/{timestamp}.txt").write_text(json.dumps({
            "timestamp": timestamp,
            "question": question,
            "category": category,
            "answer": answer
        }, indent=2))
        
        return {"answer": answer}
    except Exception as e:
        return {"answer": str(e)}
    