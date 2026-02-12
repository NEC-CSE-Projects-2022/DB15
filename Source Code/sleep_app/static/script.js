document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const resultDiv = document.getElementById('result');
    const overlay = document.querySelector('.overlay');
    
    resultDiv.querySelector('.prediction-card').innerHTML = '<div class="prediction-label">Processing...</div>';
    resultDiv.classList.add('show');
    overlay.classList.add('show');
    
    try {
        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());
        
        // Validate inputs
        for (const [key, value] of Object.entries(data)) {
            if (!value || isNaN(parseFloat(value))) {
                throw new Error(`Invalid value for ${key}`);
            }
            data[key] = parseFloat(value);
        }

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        if (result.error) {
            resultDiv.querySelector('.prediction-card').innerHTML = `
                <div class="prediction-card prediction-error">
                    <div class="prediction-label">Error</div>
                    <div class="prediction-value">${result.error}</div>
                </div>
            `;
        } else {
            resultDiv.querySelector('.prediction-card').innerHTML = `
                <div class="prediction-value">${result.prediction}</div>
                <div class="confidence-value">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
            `;
        }
    } catch (error) {
        resultDiv.querySelector('.prediction-card').innerHTML = `
            <div class="prediction-card prediction-error">
                <div class="prediction-label">Error</div>
                <div class="prediction-value">${error.message}</div>
            </div>
        `;
    }
});

// Update close handlers
document.querySelector('.close-button').addEventListener('click', function() {
    document.getElementById('result').classList.remove('show');
    document.querySelector('.overlay').classList.remove('show');
});

document.querySelector('.overlay').addEventListener('click', function() {
    document.getElementById('result').classList.remove('show');
    this.classList.remove('show');
});