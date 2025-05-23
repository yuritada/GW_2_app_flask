// import React, { useState } from 'react';
// import './App.css';

// function App() {
//   const [inputFeatures, setInputFeatures] = useState<string>(''); // 入力文字列
//   const [prediction, setPrediction] = useState<any>(null); // 予測結果
//   const [loading, setLoading] = useState<boolean>(false); // ローディング状態
//   const [error, setError] = useState<string | null>(null); // エラーメッセージ

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     setLoading(true);
//     setError(null);

//     try {
//       // 入力文字列をカンマで分割して数値配列に変換
//       const features = inputFeatures.split(',').map(Number);
      
//       // バリデーション
//       if (features.some(isNaN)) {
//         throw new Error('入力は数値をカンマ区切りで入力してください');
//       }

//       // APIリクエスト
//       const response = await fetch('http://localhost:8000/predict', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ features }),
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         throw new Error(errorData.detail || '予測に失敗しました');
//       }

//       const result = await response.json();
//       setPrediction(result);
//     } catch (err: any) {
//       setError(err.message);
//       setPrediction(null);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="App">
//       <header className="App-header">
//         <h1>機械学習モデル予測アプリ</h1>
        
//         <form onSubmit={handleSubmit}>
//           <div>
//             <label>
//               特徴量 (カンマ区切り):
//               <input
//                 type="text"
//                 value={inputFeatures}
//                 onChange={(e) => setInputFeatures(e.target.value)}
//                 placeholder="5.1,3.5,1.4,0.2"
//               />
//             </label>
//           </div>
//           <button type="submit" disabled={loading}>
//             {loading ? '予測中...' : '予測する'}
//           </button>
//         </form>

//         {error && <div className="error">{error}</div>}

//         {prediction && (
//           <div className="results">
//             <h2>予測結果:</h2>
//             <p>クラス: {prediction.prediction}</p>
//             {prediction.probability && (
//               <div>
//                 <h3>確率:</h3>
//                 <pre>{JSON.stringify(prediction.probability, null, 2)}</pre>
//               </div>
//             )}
//           </div>
//         )}
//       </header>
//     </div>
//   );
// }

// export default App;

import React, { useState } from 'react';
import './App.css';

// 投球タイプのオプション
const pitchTypes = [
  'ストレート', 'カーブ', 'スライダー', 'フォーク', 'シュート',
  'シンカー', 'チェンジアップ', 'カットボール', 'スプリット'
];

// 初期値の設定
const initialInput = {
  pitch_type: 'ストレート',
  velocity: 140,
  coordinate_x: 0,
  coordinate_y: 0,
  ball_count: 0,
  strike_count: 0,
  out_count: 0,
  runner_1b: 0,
  runner_2b: 0,
  runner_3b: 0,
  inning: 1,
  top_bottom: 0,
  score1: 0,
  score2: 0,
  batting_side: 1,
  pitching_side: 1,
  batting_order: 1,
  pitcher_order: 1,
  team_id_batter: 1,
  team_id_pitcher: 2,
  ground_id: 1
};

function App() {
  const [inputData, setInputData] = useState(initialInput);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // 入力フィールドの変更ハンドラ
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setInputData({
      ...inputData,
      [name]: name === 'pitch_type' ? value : Number(value)
    });
  };

  // フォーム送信ハンドラ
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // 追加の特徴量を計算してリクエストデータを作成
      const requestData = {
        ...inputData,
        // 基本的な特徴だけをAPIに送信し、残りはバックエンド側で計算
      };

      // APIリクエスト
      const response = await fetch('http://localhost:8000/predict_foul', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
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

  // 数値入力フィールドのレンダリング
  const renderNumberInput = (
    label: string,
    name: keyof typeof inputData,
    min?: number,
    max?: number,
    step?: number
  ) => (
    <div className="form-group">
      <label>
        {label}:
        <input
          type="number"
          name={name}
          value={inputData[name]}
          onChange={handleChange}
          min={min}
          max={max}
          step={step || 1}
        />
      </label>
    </div>
  );

  // ラジオボタンのレンダリング
  // const renderRadioGroup = (
  //   label: string,
  //   name: keyof typeof inputData,
  //   options: { value: number; label: string }[]
  // ) => (
  //   <div className="form-group">
  //     <label>{label}:</label>
  //     <div className="radio-group">
  //       {options.map(option => (
  //         <label key={option.value}>
  //           <input
  //             type="radio"
  //             name={name}
  //             value={option.value}
  //             checked={inputData[name] === option.value}
  //             onChange={handleChange}
  //           />
  //           {option.label}
  //         </label>
  //       ))}
  //     </div>
  //   </div>
  // );


  const renderRadioGroup = (
    label: string,
    name: keyof typeof inputData,
    options: { value: number; label: string }[]
  ) => (
    <div className="form-group">
      <div className="field-label">{label}:</div> 
      <div className="radio-group">
        {options.map(option => (
          <label key={option.value}>
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={inputData[name] === option.value}
              onChange={handleChange}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </div>
    </div>
  );

  return (
    <div className="App">
      <header className="App-header">
        <h1>野球投球のファウル確率予測</h1>
        
        <form onSubmit={handleSubmit} className="prediction-form">
          <div className="form-section">
            <h2>投球基本情報</h2>
            
            <div className="form-group">
              <label>
                投球タイプ:
                <select
                  name="pitch_type"
                  value={inputData.pitch_type}
                  onChange={handleChange}
                >
                  {pitchTypes.map(type => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            
            {renderNumberInput('球速 (km/h)', 'velocity', 100, 170)}
            {renderNumberInput('横座標', 'coordinate_x', -1, 1, 0.01)}
            {renderNumberInput('縦座標', 'coordinate_y', -1, 1, 0.01)}
          </div>

          <div className="form-section">
            <h2>カウント状況</h2>
            {renderNumberInput('ボールカウント', 'ball_count', 0, 3)}
            {renderNumberInput('ストライクカウント', 'strike_count', 0, 2)}
            {renderNumberInput('アウトカウント', 'out_count', 0, 2)}
          </div>

          <div className="form-section">
            <h2>走者状況</h2>
            {renderRadioGroup('1塁走者', 'runner_1b', [
              { value: 0, label: 'なし' },
              { value: 1, label: 'あり' }
            ])}
            {renderRadioGroup('2塁走者', 'runner_2b', [
              { value: 0, label: 'なし' },
              { value: 1, label: 'あり' }
            ])}
            {renderRadioGroup('3塁走者', 'runner_3b', [
              { value: 0, label: 'なし' },
              { value: 1, label: 'あり' }
            ])}
          </div>

          <div className="form-section">
            <h2>試合状況</h2>
            {renderNumberInput('イニング', 'inning', 1, 12)}
            {renderRadioGroup('表/裏', 'top_bottom', [
              { value: 0, label: '表' },
              { value: 1, label: '裏' }
            ])}
            {renderNumberInput('チーム1得点', 'score1', 0, 30)}
            {renderNumberInput('チーム2得点', 'score2', 0, 30)}
          </div>

          <div className="form-section">
            <h2>選手情報</h2>
            {renderRadioGroup('打者の打席方向', 'batting_side', [
              { value: 0, label: '左打ち' },
              { value: 1, label: '右打ち' }
            ])}
            {renderRadioGroup('投手の投球腕', 'pitching_side', [
              { value: 0, label: '左投げ' },
              { value: 1, label: '右投げ' }
            ])}
            {renderNumberInput('打順', 'batting_order', 1, 9)}
            {renderNumberInput('投手の登板順', 'pitcher_order', 1, 10)}
          </div>

          <div className="form-section">
            <h2>チーム・場所情報</h2>
            {renderNumberInput('打者チームID', 'team_id_batter', 1, 12)}
            {renderNumberInput('投手チームID', 'team_id_pitcher', 1, 12)}
            {renderNumberInput('球場ID', 'ground_id', 1, 30)}
          </div>

          <button type="submit" disabled={loading} className="submit-button">
            {loading ? '予測中...' : 'ファウル確率を予測する'}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {prediction && (
          <div className="prediction-result">
            <h2>予測結果:</h2>
            <div className="probability-container">
              <div className="probability-bar">
                <div 
                  className="probability-fill" 
                  style={{ width: `${(prediction.foul_probability * 100000).toFixed(2)}%` }}
                ></div>
              </div>
              <p className="probability-value">
                ファウル確率: {(prediction.foul_probability * 10000).toFixed(5)}%
              </p>
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;