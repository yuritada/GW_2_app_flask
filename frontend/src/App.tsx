import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputFeatures, setInputFeatures] = useState<string>(''); // 入力文字列
  const [prediction, setPrediction] = useState<any>(null); // 予測結果
  const [loading, setLoading] = useState<boolean>(false); // ローディング状態
  const [error, setError] = useState<string | null>(null); // エラーメッセージ

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // 入力文字列をカンマで分割して数値配列に変換
      const features = inputFeatures.split(',').map(Number);
      
      // バリデーション
      if (features.some(isNaN)) {
        throw new Error('入力は数値をカンマ区切りで入力してください');
      }

      // APIリクエスト
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '予測に失敗しました');
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err: any) {
      setError(err.message);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>機械学習モデル予測アプリ</h1>
        
        <form onSubmit={handleSubmit}>
          <div>
            <label>
              特徴量 (カンマ区切り):
              <input
                type="text"
                value={inputFeatures}
                onChange={(e) => setInputFeatures(e.target.value)}
                placeholder="5.1,3.5,1.4,0.2"
              />
            </label>
          </div>
          <button type="submit" disabled={loading}>
            {loading ? '予測中...' : '予測する'}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {prediction && (
          <div className="results">
            <h2>予測結果:</h2>
            <p>クラス: {prediction.prediction}</p>
            {prediction.probability && (
              <div>
                <h3>確率:</h3>
                <pre>{JSON.stringify(prediction.probability, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;