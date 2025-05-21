# # app.py
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pickle
# import numpy as np
# import uvicorn
# import joblib

# app = FastAPI()

# # CORSを有効化（Reactアプリからのリクエストを許可）
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3002"],  # Reactアプリのオリジン
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 学習済みモデルの読み込み
# try:
#     import joblib
#     model = joblib.load('foul_prediction_model.pkl')
#     label_encoders = joblib.load('label_encoders.pkl')
#     print("モデルとラベルエンコーダーを正常に読み込みました")
# except Exception as e:
#     print(f"モデル読み込みエラー: {e}")
#     model = None
#     label_encoders = None

# # リクエストのデータ型定義
# class PredictionRequest(BaseModel):
#     features: list[float]

# # 予測API
# @app.post("/predict")
# async def predict(request: PredictionRequest):
#     if model is None:
#         raise HTTPException(status_code=500, detail="モデルが読み込めませんでした")
    
#     try:
#         # 入力データの整形
#         features = np.array(request.features).reshape(1, -1)
        
#         # 予測の実行
#         prediction = model.predict(features)
#         prediction_proba = None
        
#         # 確率も取得できる場合
#         if hasattr(model, 'predict_proba'):
#             prediction_proba = model.predict_proba(features).tolist()
        
#         return {
#             "prediction": prediction.tolist(),
#             "probability": prediction_proba
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"予測エラー: {str(e)}")

# # API情報
# @app.get("/")
# async def root():
#     return {"message": "機械学習モデルAPI", "status": "稼働中"}

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
import joblib
from typing import Optional
from math import sqrt
from datetime import datetime

app = FastAPI()

# CORSを有効化（Reactアプリからのリクエストを許可）
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:3002"],  # Reactアプリのオリジン
    allow_credentials=True,
    allow_headers=["*"],

    allow_origins=["*"],  # すべてのオリジンを許可（開発環境用）

    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # OPTIONメソッドを明示的に許可S

    expose_headers=["*"],
    max_age=86400,  # OPTIONSリクエストのキャッシュ時間（秒）
)


# 学習済みモデルの読み込み
try:
    model = joblib.load('foul_prediction_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    print("モデルとラベルエンコーダーを正常に読み込みました")
except Exception as e:
    print(f"モデル読み込みエラー: {e}")
    model = None
    label_encoders = None

# 既存のリクエスト型定義
class PredictionRequest(BaseModel):
    features: list[float]

# 野球投球データのリクエスト型
class PitchData(BaseModel):
    pitch_type: str
    velocity: float
    coordinate_x: float
    coordinate_y: float
    ball_count: int
    strike_count: int
    out_count: int
    runner_1b: int
    runner_2b: int
    runner_3b: int
    inning: int
    top_bottom: int
    score1: int
    score2: int
    batting_side: int
    pitching_side: int
    batting_order: Optional[int] = 5
    pitcher_order: Optional[int] = 1
    batter_number: Optional[int] = 15
    team_id_batter: Optional[int] = 1
    team_id_pitcher: Optional[int] = 2
    ground_id: Optional[int] = 1
    total_pitch_count: Optional[int] = 50
    batter_pitch_count: Optional[int] = 3
    series_game_number: Optional[int] = 1
    is_same_team: Optional[int] = 0
    day_of_week: Optional[int] = None
    month: Optional[int] = None
    is_weekend: Optional[int] = None

# 既存の予測API
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

# ファウル確率予測API
@app.post("/predict_foul")
async def predict_foul_endpoint(pitch_data: PitchData):
    if model is None or label_encoders is None:
        raise HTTPException(status_code=500, detail="モデルまたはエンコーダーが読み込めませんでした")
    
    try:
        # PitchDataオブジェクトを辞書に変換
        sample_input = pitch_data.dict()
        
        # ファウル確率を予測
        foul_probability = predict_foul(model, label_encoders, sample_input)
        
        return {
            "foul_probability": float(foul_probability),
            "input_summary": {
                "pitch_type": sample_input["pitch_type"],
                "velocity": sample_input["velocity"],
                "count": f"B{sample_input['ball_count']}-S{sample_input['strike_count']}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ファウル予測エラー: {str(e)}")

# Python側のpredict_foul関数をバックエンドに移植
def predict_foul(model, label_encoders, sample_input):
    """
    モデルを使用してファウルの可能性を予測する関数
    
    Args:
        model: 訓練されたLightGBMモデル（46の特徴量で訓練されている）
        label_encoders: カテゴリカル変数のエンコーダー辞書
        sample_input: 予測したい投球のデータ
    
    Returns:
        float: ファウルの可能性
    """
    import pandas as pd
    import numpy as np
    from math import sqrt
    
    # 必要な特徴量リスト（モデルがトレーニングされた順序と同じ）
    features = [
        # 投球の基本情報
        'pitch_type_encoded',     # 投球タイプ（エンコード済み）
        'velocity',               # 球速
        'coordinate_x',           # 横座標
        'coordinate_y',           # 縦座標
        'distance_from_center',   # ストライクゾーン中心からの距離
        'x_from_center',          # 中心からの水平距離
        'y_from_center',          # 中心からの垂直距離
        'pitch_quadrant',         # 投球の象限
        
        # カウント状況
        'ball_count',             # ボールカウント
        'strike_count',           # ストライクカウント
        'out_count',              # アウトカウント
        'count_encoded',          # カウント状況のエンコード値
        'total_pitch_count',      # 総投球数
        'batter_pitch_count',     # 打者への投球数
        'is_first_pitch',         # 初球かどうか
        'is_two_strikes',         # ツーストライクかどうか
        'is_three_balls',         # スリーボールかどうか
        'is_full_count',          # フルカウントかどうか
        
        # 走者状況
        'runner_1b_exists',       # 1塁走者の有無
        'runner_2b_exists',       # 2塁走者の有無
        'runner_3b_exists',       # 3塁走者の有無
        'runners_on_base',        # 走者の合計数
        'scoring_position',       # 得点圏に走者がいるか
        
        # 試合状況
        'inning',                 # イニング
        'top_bottom',             # 表裏
        'score_diff',             # スコア差
        'absolute_score_diff',    # スコア差の絶対値
        'is_close_game',          # 接戦かどうか
        'is_tie_game',            # 同点かどうか
        'game_progress',          # 試合の進行度
        'total_score',            # 合計得点
        'inning_stage',           # イニングの段階
        'series_game_number',     # シリーズの何試合目か
        
        # 打者・投手情報
        'batting_side_batter_encoded',  # 打者の打席方向（エンコード済み）
        'pitching_side_pitcher_encoded', # 投手の投球腕（エンコード済み）
        'batting_order',          # 打順
        'pitcher_order',          # 投手の登板順
        'batter_number',          # 打者番号
        
        # チーム・場所情報
        'team_id_batter',         # 打者のチームID 
        'team_id_pitcher',        # 投手のチームID
        'ground_id',              # 球場ID
        'is_same_team',           # 投手と打者が同じチームか
        'is_home_team_batting',   # ホームチームの攻撃かどうか
        
        # 時間的特徴
        'day_of_week',            # 曜日（0=月曜, 6=日曜）
        'month',                  # 月
        'is_weekend',             # 週末フラグ
    ]
    
    # 特徴量の辞書を初期化
    feature_dict = {}
    
    # 1. 投球の基本情報
    if 'pitch_type' in sample_input:
        feature_dict['pitch_type_encoded'] = label_encoders['pitch_type'].transform([sample_input['pitch_type']])[0]
    else:
        feature_dict['pitch_type_encoded'] = 0
    
    feature_dict['velocity'] = sample_input.get('velocity', 140)
    feature_dict['coordinate_x'] = sample_input.get('coordinate_x', 0)
    feature_dict['coordinate_y'] = sample_input.get('coordinate_y', 0)
    
    # ストライクゾーン中心からの距離を計算
    x = feature_dict['coordinate_x']
    y = feature_dict['coordinate_y']
    feature_dict['distance_from_center'] = sqrt(x**2 + y**2)
    feature_dict['x_from_center'] = x
    feature_dict['y_from_center'] = y
    
    # 投球の象限
    if x >= 0 and y >= 0:
        feature_dict['pitch_quadrant'] = 1
    elif x < 0 and y >= 0:
        feature_dict['pitch_quadrant'] = 2
    elif x < 0 and y < 0:
        feature_dict['pitch_quadrant'] = 3
    else:
        feature_dict['pitch_quadrant'] = 4
    
    # 2. カウント状況
    feature_dict['ball_count'] = sample_input.get('ball_count', 0)
    feature_dict['strike_count'] = sample_input.get('strike_count', 0)
    feature_dict['out_count'] = sample_input.get('out_count', 0)
    feature_dict['count_encoded'] = feature_dict['ball_count'] * 10 + feature_dict['strike_count']
    
    # 投球数関連（仮の値または入力値）
    feature_dict['total_pitch_count'] = sample_input.get('total_pitch_count', 50)
    feature_dict['batter_pitch_count'] = sample_input.get('batter_pitch_count', 3)
    
    # カウント状況のフラグ
    feature_dict['is_first_pitch'] = 1 if feature_dict['ball_count'] == 0 and feature_dict['strike_count'] == 0 else 0
    feature_dict['is_two_strikes'] = 1 if feature_dict['strike_count'] == 2 else 0
    feature_dict['is_three_balls'] = 1 if feature_dict['ball_count'] == 3 else 0
    feature_dict['is_full_count'] = 1 if feature_dict['ball_count'] == 3 and feature_dict['strike_count'] == 2 else 0
    
    # 3. 走者状況
    feature_dict['runner_1b_exists'] = sample_input.get('runner_1b', 0)
    feature_dict['runner_2b_exists'] = sample_input.get('runner_2b', 0)
    feature_dict['runner_3b_exists'] = sample_input.get('runner_3b', 0)
    feature_dict['runners_on_base'] = (
        feature_dict['runner_1b_exists'] + 
        feature_dict['runner_2b_exists'] + 
        feature_dict['runner_3b_exists']
    )
    feature_dict['scoring_position'] = 1 if feature_dict['runner_2b_exists'] or feature_dict['runner_3b_exists'] else 0
    
    # 4. 試合状況
    feature_dict['inning'] = sample_input.get('inning', 1)
    feature_dict['top_bottom'] = sample_input.get('top_bottom', 0)
    
    # スコア関連
    score1 = sample_input.get('score1', 0)
    score2 = sample_input.get('score2', 0)
    if feature_dict['top_bottom'] == 0:  # 表の場合
        feature_dict['score_diff'] = score1 - score2
    else:  # 裏の場合
        feature_dict['score_diff'] = score2 - score1
    
    feature_dict['absolute_score_diff'] = abs(feature_dict['score_diff'])
    feature_dict['is_close_game'] = 1 if feature_dict['absolute_score_diff'] <= 3 else 0
    feature_dict['is_tie_game'] = 1 if feature_dict['score_diff'] == 0 else 0
    feature_dict['total_score'] = score1 + score2
    
    # 試合進行状況
    feature_dict['game_progress'] = (feature_dict['inning'] - 1 + feature_dict['top_bottom'] * 0.5) / 9.0
    
    # イニングの段階
    inning = feature_dict['inning']
    if inning <= 3:
        feature_dict['inning_stage'] = 0  # 序盤
    elif inning <= 6:
        feature_dict['inning_stage'] = 1  # 中盤
    else:
        feature_dict['inning_stage'] = 2  # 終盤
    
    # シリーズの試合番号
    feature_dict['series_game_number'] = sample_input.get('series_game_number', 1)
    
    # 5. 打者・投手情報
    feature_dict['batting_side_batter_encoded'] = int(sample_input.get('batting_side', 1))
    feature_dict['pitching_side_pitcher_encoded'] = int(sample_input.get('pitching_side', 1))
    feature_dict['batting_order'] = sample_input.get('batting_order', 5)
    feature_dict['pitcher_order'] = sample_input.get('pitcher_order', 1)
    feature_dict['batter_number'] = sample_input.get('batter_number', 15)
    
    # 6. チーム・場所情報
    feature_dict['team_id_batter'] = sample_input.get('team_id_batter', 1)
    feature_dict['team_id_pitcher'] = sample_input.get('team_id_pitcher', 2)
    feature_dict['ground_id'] = sample_input.get('ground_id', 3)
    feature_dict['is_same_team'] = sample_input.get('is_same_team', 0)
    feature_dict['is_home_team_batting'] = 1 if feature_dict['top_bottom'] == 1 else 0
    
    # 7. 時間的特徴（現在の日付から取得または指定された値を使用）
    current_date = datetime.now()
    feature_dict['day_of_week'] = sample_input.get('day_of_week', current_date.weekday())
    feature_dict['month'] = sample_input.get('month', current_date.month)
    feature_dict['is_weekend'] = 1 if feature_dict['day_of_week'] >= 5 else 0
    
    # 特徴量辞書からデータフレームを作成
    input_df = pd.DataFrame([feature_dict])
    
    # 特徴量の順序を確認
    for feature in features:
        if feature not in input_df.columns:
            raise ValueError(f"特徴量 '{feature}' がデータフレームにありません。")
    
    # 特徴量の順序を揃える
    input_df = input_df[features]
    
    # 特徴量の数を確認
    if len(input_df.columns) != 46:
        raise ValueError(f"特徴量の数が46ではありません。現在の特徴量数: {len(input_df.columns)}")
    
    # ファウルの可能性を予測
    foul_probability = model.predict_proba(input_df)[0, 1]
    
    return foul_probability

# API情報
@app.get("/")
async def root():
    return {
        "message": "野球投球ファウル予測API",
        "status": "稼働中",
        "endpoints": {
            "/predict": "一般的な機械学習予測",
            "/predict_foul": "野球投球のファウル確率予測"
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)