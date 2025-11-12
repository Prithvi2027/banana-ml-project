const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const analyzing = document.getElementById('analyzing');
const results = document.getElementById('results');
const resetBtn = document.getElementById('resetBtn');

// Upload area interactions
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImage(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImage(file);
    }
});

resetBtn.addEventListener('click', () => {
    previewContainer.style.display = 'none';
    analyzing.style.display = 'none';
    results.style.display = 'none';
    uploadArea.style.display = 'block';
    fileInput.value = '';
});

async function handleImage(file) {
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Show analyzing state
    analyzing.style.display = 'block';

    // Send to API
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image. Please try another photo.');
        resetBtn.click();
    }
}

function displayResults(data) {
    // Hide analyzing
    analyzing.style.display = 'none';

    // Update stage and emoji
    const stageText = `${data.emoji} ${data.class.charAt(0).toUpperCase() + data.class.slice(1)}`;
    document.getElementById('ripenessStage').textContent = stageText;
    
    // Update days left
    document.getElementById('daysLeft').textContent = data.display_days;
    
    // Update details
    document.getElementById('classification').textContent = data.class.charAt(0).toUpperCase() + data.class.slice(1);
    document.getElementById('confidence').textContent = `${data.confidence}%`;
    document.getElementById('recommendation').textContent = data.recommendation;

    // Update progress bar based on ripeness
    const ripenessMap = { 'unripe': 25, 'ripe': 50, 'overripe': 75, 'rotten': 100 };
    const progressValue = ripenessMap[data.class] || 50;
    const progressFill = document.getElementById('progressFill');
    progressFill.style.width = '0%';
    setTimeout(() => {
        progressFill.style.width = `${progressValue}%`;
    }, 100);

    // Display class probabilities
    displayProbabilities(data.probabilities);

    // Show results
    results.style.display = 'block';

    // Emoji celebration
    createEmojiRain(data.emoji);
}

function displayProbabilities(probabilities) {
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';

    // Sort by probability (descending)
    const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([className, prob]) => {
        const container = document.createElement('div');
        container.className = 'prob-bar-container';

        const label = document.createElement('div');
        label.className = 'prob-label';
        label.innerHTML = `
            <span class="prob-name">${className}</span>
            <span class="prob-value">${prob.toFixed(1)}%</span>
        `;

        const barBg = document.createElement('div');
        barBg.className = 'prob-bar-bg';

        const barFill = document.createElement('div');
        barFill.className = 'prob-bar-fill';
        barFill.style.width = '0%';
        
        barBg.appendChild(barFill);
        container.appendChild(label);
        container.appendChild(barBg);
        probBars.appendChild(container);

        // Animate after a short delay
        setTimeout(() => {
            barFill.style.width = `${prob}%`;
        }, 100);
    });
}

function createEmojiRain(emoji) {
    for (let i = 0; i < 15; i++) {
        setTimeout(() => {
            const emojiEl = document.createElement('div');
            emojiEl.className = 'falling-emoji';
            emojiEl.textContent = emoji;
            emojiEl.style.left = Math.random() * 100 + '%';
            emojiEl.style.animationDuration = (2 + Math.random() * 2) + 's';
            document.body.appendChild(emojiEl);

            setTimeout(() => emojiEl.remove(), 3000);
        }, i * 100);
    }
}

// Add welcome animation
window.addEventListener('load', () => {
    document.querySelector('.container').style.animation = 'fadeIn 0.5s ease';
});
