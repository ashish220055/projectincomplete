import React, { useState, useEffect } from 'react';

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [health, setHealth] = useState("Connecting to Backend...");
  const [isOnline, setIsOnline] = useState(false);
  
  const [regime, setRegime] = useState('--');
  const [prediction, setPrediction] = useState(0.0);
  const [directionMsg, setDirectionMsg] = useState("Awaiting sequence data...");

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) {
          setHealth("Backend Connected");
          setIsOnline(true);
        }
      } catch (error) {
        setHealth("Backend Offline - Run 'uvicorn src.api.main:app'");
        setIsOnline(false);
      }
    };
    
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let pollingInterval;
    if (isLiveMode) {
      setDirectionMsg("Connecting to Binance stream (needs 60m buffer)...");
      pollingInterval = setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/predict/latest`);
          if (res.ok) {
            const data = await res.json();
            if (data.timestamp > 0) {
              setRegime(data.regime);
              setPrediction(data.prediction);
              setDirectionMsg(data.prediction > 0 ? "Bullish Outlook (Live)" : "Bearish Outlook (Live)");
            }
          }
        } catch (err) {
          console.error("Polling Error", err);
        }
      }, 2000);
    } else {
      setDirectionMsg("Stream disconnected.");
    }
    return () => clearInterval(pollingInterval);
  }, [isLiveMode]);

  const percentStr = (prediction * 100).toFixed(4);
  const isPositive = prediction > 0;

  return (
    <>
      <div className="background-glow"></div>
      
      <div className="container">
        <header>
          <h1>Market Microstructure <span>AI</span></h1>
          <p>Deep Learning Pattern Discovery & Baseline Prediction</p>
          <div className="status-indicator">
            <div className={`pulse-dot ${isOnline ? 'online' : ''}`}></div>
            <span>{health}</span>
          </div>
        </header>

        <main className="dashboard">
          {/* Regime Card */}
          <div className="card glassmorphism">
            <div className="card-header">
              <h2>Latent Market Regime</h2>
              <span className="badge autoencoder-badge">Autoencoder</span>
            </div>
            <div className="card-body">
              <div className="regime-display">
                <div className={`regime-circle r${regime !== '--' ? regime : '0'}`}>
                  <span>{regime}</span>
                </div>
              </div>
              <p className="description">Current identified structural state based on 60-minute order flow latent embeddings.</p>
            </div>
          </div>

          {/* Prediction Card */}
          <div className="card glassmorphism">
            <div className="card-header">
              <h2>Predicted Next Return</h2>
              <span className="badge lstm-badge">LSTM Predictor</span>
            </div>
            <div className="card-body">
              <div className="prediction-display">
                <h1 className={isPositive ? 'positive' : 'negative'}>
                  {isPositive ? '+' : ''}{percentStr}%
                </h1>
                <p className="direction">{directionMsg}</p>
              </div>
              <p className="description">Baseline forward step prediction using supervised sequence modeling.</p>
            </div>
          </div>
        </main>

        <div className="action-section">
          <button 
            className={`glow-button ${isLiveMode ? 'stop' : ''}`}
            onClick={() => setIsLiveMode(!isLiveMode)}
          >
            {isLiveMode ? "Stop Live Feed" : "Connect to Live Data Feed"}
          </button>
        </div>
      </div>
    </>
  );
}

export default App;
