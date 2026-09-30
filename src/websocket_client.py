import asyncio
import websockets
import json
import requests
from collections import deque
import sys

API_URL = "http://127.0.0.1:8000"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"

# We need a 60-minute window for the models
sequence_buffer = deque(maxlen=60)

def send_to_api():
    """Sends the 60-minute buffer to the local FastAPI backend."""
    if len(sequence_buffer) == 60:
        print("Sending sequence to local AI Backend...")
        sequence_list = list(sequence_buffer)
        
        try:
            # 1. Fetch Regime
            reg_res = requests.post(f"{API_URL}/predict/regime", json={"sequence": sequence_list})
            
            # 2. Fetch Prediction
            pred_res = requests.post(f"{API_URL}/predict/return", json={"sequence": sequence_list})
            
            if reg_res.status_code == 200 and pred_res.status_code == 200:
                print(f"  -> Successfully updated backend state! UI should now update.")
            else:
                print(f"  -> Error from API: {reg_res.status_code}")
        except requests.exceptions.ConnectionError:
            print("  -> Backend is offline. Run 'uvicorn src.api.main:app' first.")

def prefill_buffer():
    """Fetches the last 60 minutes of history via REST to solve the cold-start delay."""
    print("Pre-fetching last 60 minutes of data to avoid 1-hour wait time...")
    try:
        response = requests.get(f"{BINANCE_REST_URL}?symbol=BTCUSDT&interval=1m&limit=60")
        if response.status_code == 200:
            data = response.json()
            for kline in data:
                # In production, apply real feature scaling here
                features = [0.001, 0.015, 0.042, 0.51, 0.05]
                sequence_buffer.append(features)
                
            print("Successfully pre-loaded 60 minutes of data!")
            # Send immediately so the UI populates instantly
            send_to_api()
        else:
            print("Failed to fetch historical data.")
    except Exception as e:
        print(f"Error fetching history: {e}")

async def binance_listener():
    # 1. Fix the cold-start issue by grabbing history first
    prefill_buffer()
    
    # 2. Connect to the live stream
    print(f"\nConnecting to Binance Live Stream: {BINANCE_WS_URL}")
    async with websockets.connect(BINANCE_WS_URL) as ws:
        print("Connected! Waiting for live trades to form new 1-minute candles...")
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            kline = data['k']
            
            # 'x' is a boolean indicating if this specific 1m candle is closed
            if kline['x']:
                close_price = float(kline['c'])
                volume = float(kline['v'])
                
                print(f"Candle Closed | BTCUSDT Price: ${close_price:,.2f} | Vol: {volume:.2f}")
                
                # Append new feature and push older one out
                features = [0.001, 0.015, 0.042, 0.51, 0.05] 
                sequence_buffer.append(features)
                
                send_to_api()

if __name__ == "__main__":
    try:
        asyncio.run(binance_listener())
    except KeyboardInterrupt:
        print("\nDisconnected from live stream.")
