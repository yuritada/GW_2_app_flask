# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

app = FastAPI()

# CORSを有効化（Reactアプリからのリクエストを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Reactアプリのオリジン
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 学習済みモデルの読み込み
try:
    with open('foul_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("モデルを正常に読み込みました")
except Exception as e:
    print(f"モデル読み込みエラー: {e}")
    model = None

# リクエストのデータ型定義
class PredictionRequest(BaseModel):
    features: list[float]

# 予測API
@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="モデルが読み込めませんでした")
    
    try:
        # 入力データの整形
        features = np.array(request.features).reshape(1, -1)
        
        # 予測の実行
        prediction = model.predict(features)
        prediction_proba = None
        
        # 確率も取得できる場合
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features).tolist()
        
        return {
            "prediction": prediction.tolist(),
            "probability": prediction_proba
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"予測エラー: {str(e)}")

# API情報
@app.get("/")
async def root():
    return {"message": "機械学習モデルAPI", "status": "稼働中"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)