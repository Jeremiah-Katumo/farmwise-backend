from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import SessionLocal, engine, Base, get_db
from joblib import load
import pandas as pd
from .routes import routes
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import openai

load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def root():
    return {
        "message": "Welcome to the AgroAI API",
        "status": "running"
    }

app.include_router(routes.router)
