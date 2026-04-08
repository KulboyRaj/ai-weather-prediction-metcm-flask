let currentHourlyData = [];
let currentHourlyMetcm = [];

function getPrediction() {
    const datePicker = document.getElementById('datePicker');
    const predictBtn = document.getElementById('predictBtn');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const results = document.getElementById('results');

    const selectedDate = datePicker.value;

    if (!selectedDate) {
        showError('Please select a date');
        return;
    }

    // Show loading, hide other sections
    loading.classList.remove('hidden');
    error.classList.add('hidden');
    results.classList.add('hidden');
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ date: selectedDate })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        loading.classList.add('hidden');
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-bolt"></i> Predict Weather';

        if (data.success) {
            currentHourlyData = data.hourly_data || [];
            currentHourlyMetcm = data.hourly_metcm || [];
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    })
    .catch(error => {
        loading.classList.add('hidden');
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-bolt"></i> Predict Weather';
        showError('Network error: ' + error.message);
        console.error('Prediction error:', error);
    });
}

function displayResults(data) {
    const results = document.getElementById('results');
    const summary = data.summary;

    // Update summary
    document.getElementById('summaryDate').textContent =
        `Weather Forecast for ${new Date(data.date).toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        })}`;

    document.getElementById('maxTemp').textContent = summary.max_temp;
    document.getElementById('minTemp').textContent = summary.min_temp;
    document.getElementById('avgHumidity').textContent = summary.avg_humidity;
    document.getElementById('maxWind').textContent = summary.max_wind;
    document.getElementById('totalPrecip').textContent = summary.total_precip;
    document.getElementById('dominantCondition').textContent = summary.dominant_condition;

    // Store hourly data
    currentHourlyData = data.hourly_data || [];
    currentHourlyMetcm = data.hourly_metcm || [];

    // Initialize slider for 12 time slots
    const slider = document.getElementById('hourSlider');
    slider.value = 6; // Default to 12:00
    updateHourlyDisplay(6);

    // ✅ INITIALIZE COPY BUTTON HERE (after results are shown)
    initializeCopyButton();

    // Show results with animation
    results.classList.remove('hidden');
    results.style.opacity = '0';
    results.style.transform = 'translateY(20px)';

    setTimeout(() => {
        results.style.transition = 'all 0.5s ease';
        results.style.opacity = '1';
        results.style.transform = 'translateY(0)';
    }, 100);
}

function updateHourlyDisplay(slot) {
    if (!currentHourlyData[slot]) {
        console.log('No data for slot:', slot);
        return;
    }

    const data = currentHourlyData[slot];
    const weatherCard = document.querySelector('.weather-card');

    // Add animation to weather card
    weatherCard.style.transform = 'scale(0.95)';
    weatherCard.style.opacity = '0.8';

    // Update weather icon based on condition
    const iconClass = getWeatherIcon(data.condition);
    document.getElementById('weatherIcon').innerHTML = `<i class="fas ${iconClass}"></i>`;

    // Update all weather information
    document.getElementById('currentTime').textContent = data.hour;
    document.getElementById('currentTemp').textContent = `${data.temp}°C`;
    document.getElementById('currentCondition').textContent = data.condition;
    document.getElementById('currentHumidity').textContent = `${data.humidity}%`;
    document.getElementById('currentWind').textContent = `${data.windspeed} km/h`;
    document.getElementById('currentWindDir').textContent = `${data.winddir}°`;
    document.getElementById('currentPressure').textContent = `${data.sealevelpressure} hPa`;
    document.getElementById('currentCloud').textContent = `${data.cloudcover}%`;
    document.getElementById('currentVisibility').textContent = `${data.visibility} km`;
    document.getElementById('currentPrecip').textContent = `${data.precip} mm`;
    document.getElementById('currentDew').textContent = `${data.dew}°C`;

    // Update METCM information for this specific hour
    updateMetcmForHour(slot);

    // Complete animation
    setTimeout(() => {
        weatherCard.style.transition = 'all 0.3s ease';
        weatherCard.style.transform = 'scale(1)';
        weatherCard.style.opacity = '1';
    }, 150);
}

function updateMetcmForHour(slot) {
    if (!currentHourlyMetcm[slot]) {
        console.log('No METCM data for slot:', slot);
        // Show default message or hide section
        document.getElementById('metcmCode').textContent = 'METCM data not available for this hour';
        document.getElementById('metcmExplanation').textContent = 'Artillery meteorological data will be displayed here when available.';
        return;
    }

    const metcmData = currentHourlyMetcm[slot];

    // Update METCM display with hourly data
    document.getElementById('metcmCode').textContent = metcmData.metcm;
    document.getElementById('metcmExplanation').innerHTML = metcmData.explanation.replace(/\n/g, '<br>');

    // Update time indicator
    document.getElementById('metcmTimeIndicator').textContent = `for ${metcmData.hour}`;

    // Add animation to METCM section
    const metcmSection = document.querySelector('.metcm-section');
    metcmSection.style.transform = 'scale(0.98)';
    metcmSection.style.opacity = '0.9';

    setTimeout(() => {
        metcmSection.style.transition = 'all 0.3s ease';
        metcmSection.style.transform = 'scale(1)';
        metcmSection.style.opacity = '1';
    }, 150);
}

function getWeatherIcon(condition) {
    const iconMap = {
        'Clear': 'fa-sun',
        'Mostly Clear': 'fa-sun',
        'Partly Cloudy': 'fa-cloud-sun',
        'Cloudy': 'fa-cloud',
        'Rain': 'fa-cloud-rain',
        'Snow': 'fa-snowflake',
        'Thunderstorm': 'fa-bolt',
        'Fog': 'fa-smog',
        'Mist': 'fa-smog',
        'Clouds': 'fa-cloud'  // Added for API conditions
    };
    return iconMap[condition] || 'fa-cloud';
}

function showError(message) {
    const error = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');

    errorMessage.textContent = message;
    error.classList.remove('hidden');

    // Add animation to error
    error.style.opacity = '0';
    error.style.transform = 'translateY(-10px)';

    setTimeout(() => {
        error.style.transition = 'all 0.3s ease';
        error.style.opacity = '1';
        error.style.transform = 'translateY(0)';
    }, 100);
}

// ✅ COPY FUNCTIONALITY
function copyMetcmToClipboard() {
    const metcmCode = document.getElementById('metcmCode');
    const copyBtn = document.getElementById('copyMetcmBtn');

    if (!metcmCode || !metcmCode.textContent.trim() || metcmCode.textContent === 'METCM data not available for this hour') {
        showError('No METCM data available to copy');
        return;
    }

    const metcmText = metcmCode.textContent;

    // Use the modern Clipboard API
    navigator.clipboard.writeText(metcmText).then(() => {
        // Success feedback
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.classList.add('copied');

        // Revert button after 2 seconds
        setTimeout(() => {
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            copyBtn.classList.remove('copied');
        }, 2000);

        console.log('✅ METCM message copied to clipboard');
    }).catch(err => {
        // Fallback for older browsers
        console.error('❌ Failed to copy: ', err);
        showError('Failed to copy METCM message');

        // Fallback method
        const textArea = document.createElement('textarea');
        textArea.value = metcmText;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);

        // Show success even with fallback
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.classList.add('copied');
        setTimeout(() => {
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            copyBtn.classList.remove('copied');
        }, 2000);
    });
}

// ✅ INITIALIZE COPY BUTTON (Call this after results are displayed)
function initializeCopyButton() {
    const copyBtn = document.getElementById('copyMetcmBtn');
    if (copyBtn) {
        // Remove existing event listeners to avoid duplicates
        copyBtn.replaceWith(copyBtn.cloneNode(true));

        // Get the new button reference
        const newCopyBtn = document.getElementById('copyMetcmBtn');
        newCopyBtn.addEventListener('click', copyMetcmToClipboard);
        console.log('✅ Copy button initialized');
    } else {
        console.log('❌ Copy button not found');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    const slider = document.getElementById('hourSlider');
    const datePicker = document.getElementById('datePicker');

    // Slider event with smooth updates
    let sliderTimeout;
    slider.addEventListener('input', function() {
        // Debounce the slider updates for smoother animation
        clearTimeout(sliderTimeout);
        sliderTimeout = setTimeout(() => {
            updateHourlyDisplay(parseInt(this.value));
        }, 50);
    });

    // Enter key support for date picker
    datePicker.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            getPrediction();
        }
    });

    // Auto-predict tomorrow's weather on load
    setTimeout(() => {
        console.log('Auto-predicting weather...');
        getPrediction();
    }, 1000);
});