import asyncio
import websockets
import json
import requests
from collections import deque
import sys

API_URL = "http://127.0.0.1:8000"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"

# We need a 60-minute window for the models
sequence_buffer = deque(maxlen=60)

async def binance_listener():
    print(f"Connecting to Binance Live Stream: {BINANCE_WS_URL}")
    
    async with websockets.connect(BINANCE_WS_URL) as ws:
        print("Connected! Waiting for live trades to form 1-minute candles...")
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            kline = data['k']
            
            # 'x' is a boolean indicating if this specific 1m candle is closed
            if kline['x']:
                close_price = float(kline['c'])
                volume = float(kline['v'])
                
                print(f"Candle Closed | BTCUSDT Price: ${close_price:,.2f} | Vol: {volume:.2f}")
                
                # --- FEATURE ENGINEERING PLACEHOLDER ---
                # In production, we'd apply our exact `data_pipeline.py` logic here
                # and scale the features using a saved StandardScaler.
                # For this MVP extension, we simulate the 5 extracted features.
                features = [0.001, 0.015, 0.042, 0.51, 0.05] 
                sequence_buffer.append(features)
                
                if len(sequence_buffer) == 60:
                    print("Sequence buffer full (60m). Sending to local AI Backend...")
                    sequence_list = list(sequence_buffer)
                    
                    try:
                        # 1. Fetch Regime
                        reg_res = requests.post(f"{API_URL}/predict/regime", json={"sequence": sequence_list})
                        
                        # 2. Fetch Prediction
                        pred_res = requests.post(f"{API_URL}/predict/return", json={"sequence": sequence_list})
                        
                        if reg_res.status_code == 200 and pred_res.status_code == 200:
                            print(f"  -> Successfully updated backend state!")
                        else:
                            print(f"  -> Error from API: {reg_res.status_code}")
                    except requests.exceptions.ConnectionError:
                        print("  -> Backend is offline. Run 'uvicorn src.api.main:app' first.")

if __name__ == "__main__":
    try:
        asyncio.run(binance_listener())
    except KeyboardInterrupt:
        print("\nDisconnected from live stream.")
