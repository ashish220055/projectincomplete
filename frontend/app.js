const API_URL = "http://127.0.0.1:8000";

// DOM Elements
const statusIndicator = document.getElementById("api-status");
const pulseDot = document.querySelector(".pulse-dot");
const regimeCircle = document.getElementById("regime-circle");
const regimeId = document.getElementById("regime-id");
const predValue = document.getElementById("prediction-value");
const predDirection = document.getElementById("prediction-direction");
const simulateBtn = document.getElementById("simulate-btn");

// Check API Health
async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) {
            statusIndicator.textContent = "Backend Connected";
            pulseDot.classList.add("online");
        }
    } catch (error) {
        statusIndicator.textContent = "Backend Offline - Run 'uvicorn src.api.main:app'";
        pulseDot.classList.remove("online");
    }
}

// Simulate fetching data from the API
async function simulateMarketData() {
    // Generate dummy sequence data matching (60, 5) shape
    const dummySequence = Array.from({length: 60}, () => 
        Array.from({length: 5}, () => Math.random())
    );

    simulateBtn.textContent = "Processing Neural Networks...";
    simulateBtn.style.opacity = "0.7";

    try {
        // Fetch Regime
        const regimeRes = await fetch(`${API_URL}/predict/regime`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ sequence: dummySequence })
        });
        
        // Fetch Prediction
        const predRes = await fetch(`${API_URL}/predict/return`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ sequence: dummySequence })
        });

        if(regimeRes.ok && predRes.ok) {
            const regimeData = await regimeRes.json();
            const predData = await predRes.json();
            
            updateDashboard(predData.predicted_return);
        }
    } catch (err) {
        console.error("API Error", err);
        // Fallback simulation if API isn't running yet so user can see the UI
        updateDashboard(null); 
    }

    simulateBtn.textContent = "Simulate Live Data Feed";
    simulateBtn.style.opacity = "1";
}

function updateDashboard(actualPred) {
    // Randomly select a regime 0-3 for demonstration
    const regime = Math.floor(Math.random() * 4);
    
    // Use actual pred if API is up, otherwise fake it for UI demo
    const prediction = actualPred !== null ? actualPred : (Math.random() * 0.04 - 0.02);
    
    // Update Regime UI
    regimeId.textContent = regime;
    regimeCircle.className = `regime-circle r${regime}`;

    // Update Prediction UI
    const percent = (prediction * 100).toFixed(2);
    predValue.textContent = `${percent > 0 ? '+' : ''}${percent}%`;
    
    if (prediction > 0) {
        predValue.className = "positive";
        predDirection.textContent = "Bullish Outlook";
    } else {
        predValue.className = "negative";
        predDirection.textContent = "Bearish Outlook";
    }
}

// Event Listeners
simulateBtn.addEventListener("click", simulateMarketData);

// Init
checkHealth();
// Ping health every 5 seconds
setInterval(checkHealth, 5000);
