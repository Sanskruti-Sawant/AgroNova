
document.addEventListener('DOMContentLoaded', () => {
    const translations = {
        'en': {
            'page-title': 'AgriNova | Predict',
            'nav-home': 'Home',
            'nav-predict': 'Predict',
            'nav-about': 'About us',
            'nav-support': 'Support',
            'form-title': 'Start Your Prediction',
            'form-subtitle': 'Upload a soil image and provide your location for a personalized plan.',
            'label-soil-image': 'Soil Image (JPG/PNG)',
            'label-location': 'Your Location',
            'label-npk': 'Optional NPK Values',
            'btn-location': 'Use My Current Location',
            'btn-manual-weather': 'Get Weather',
            'btn-submit': 'Get Prediction',
            'loader-text': 'Analyzing soil... this may take a moment.',
            'status-detecting': 'Detecting your location...',
            'status-not-detected': 'Location not detected.',
            'error-geolocation-failed': 'Geolocation failed:',
            'error-geolocation-unsupported': 'Geolocation is not supported by this browser.',
            'error-weather-failed': 'Failed to fetch weather data. Please try again.',
            'result-title-weather': 'Weather Details',
            'result-title-crop': 'Crop Suggestions',
            'result-title-fert': 'Fertilizer Plan',
            'result-title-explanation': 'Explanation',
            'result-metadata-time': 'Inference Time:',
            'result-metadata-version': 'Model Version:',
            'result-recommendation-prefix': 'We recommend planting',
            'result-explanation-prefix': 'Based on our analysis, your soil is primarily',
            'result-explanation-mid': 'This type of soil is well-suited for growing',
            'result-explanation-suffix': 'The recommended fertilizer,',
            'result-explanation-end': 'is suggested to provide the necessary nutrients for a healthy crop yield.',
            'weather-temp-label': 'Temperature',
            'weather-humidity-label': 'Humidity',
            'weather-moisture-label': 'Moisture',
            'weather-country-label': 'Country',
            'weather-state-label': 'State'
        },
        'hi': {
            'page-title': 'कृषि-नोवा | अनुमान',
            'nav-home': 'होम',
            'nav-predict': 'अनुमान',
            'nav-about': 'हमारे बारे में',
            'nav-support': 'सहयोग',
            'form-title': 'अपना अनुमान शुरू करें',
            'form-subtitle': 'मिट्टी की तस्वीर अपलोड करें और अपनी जगह की जानकारी दें।',
            'label-soil-image': 'मिट्टी की तस्वीर (JPG/PNG)',
            'label-location': 'आपकी जगह',
            'label-npk': 'वैकल्पिक NPK मान',
            'btn-location': 'मेरी वर्तमान जगह का उपयोग करें',
            'btn-manual-weather': 'मौसम प्राप्त करें',
            'btn-submit': 'अनुमान प्राप्त करें',
            'loader-text': 'मिट्टी का विश्लेषण हो रहा है... इसमें थोड़ा समय लग सकता है।',
            'status-detecting': 'आपकी जगह का पता लगाया जा रहा है...',
            'status-not-detected': 'जगह का पता नहीं चला।',
            'error-geolocation-failed': 'जगह की जानकारी विफल:',
            'error-geolocation-unsupported': 'आपके ब्राउज़र में जगह की जानकारी समर्थित नहीं है।',
            'error-weather-failed': 'मौसम का डेटा लाने में विफल। कृपया फिर से प्रयास करें।',
            'result-title-weather': 'मौसम का विवरण',
            'result-title-crop': 'फसल के सुझाव',
            'result-title-fert': 'खाद योजना',
            'result-title-explanation': 'स्पष्टीकरण',
            'result-metadata-time': 'अनुमान समय:',
            'result-metadata-version': 'मॉडल संस्करण:',
            'result-recommendation-prefix': 'हम लगाने की सलाह देते हैं',
            'result-explanation-prefix': 'हमारे विश्लेषण के आधार पर, आपकी मिट्टी मुख्य रूप से है',
            'result-explanation-mid': 'इस प्रकार की मिट्टी में खेती के लिए उपयुक्त है',
            'result-explanation-suffix': 'अनुशंसित उर्वरक,',
            'result-explanation-end': 'को स्वस्थ फसल के लिए आवश्यक पोषक तत्व प्रदान करने का सुझाव दिया गया है।',
            'weather-temp-label': 'तापमान',
            'weather-humidity-label': 'आर्द्रता',
            'weather-moisture-label': 'मातीतील ओलावा',
            'weather-country-label': 'देश',
            'weather-state-label': 'राज्य'
        },
        'mr': {
            'page-title': 'ॲग्रीनोव्हा | अनुमान',
            'nav-home': 'मुख्यपृष्ठ',
            'nav-predict': 'अनुमान',
            'nav-about': 'आमच्याबद्दल',
            'nav-support': 'सहाय्य',
            'form-title': 'तुमचा अनुमान सुरू करा',
            'form-subtitle': 'मातीचा फोटो अपलोड करा आणि तुमच्या स्थानाची माहिती द्या.',
            'label-soil-image': 'मातीचा फोटो (JPG/PNG)',
            'label-location': 'तुमचे स्थान',
            'label-npk': 'वैकल्पिक NPK मूल्ये',
            'btn-location': 'माझ्या वर्तमान स्थानाचा वापर करा',
            'btn-manual-weather': 'हवामान मिळवा',
            'btn-submit': 'अनुमान मिळवा',
            'loader-text': 'मातीचे विश्लेषण होत आहे... यास थोडा वेळ लागू शकतो.',
            'status-detecting': 'तुमचे स्थान शोधले जात आहे...',
            'status-not-detected': 'स्थान सापडले नाही.',
            'error-geolocation-failed': 'स्थान शोधण्यात अयशस्वी:',
            'error-geolocation-unsupported': 'तुमच्या ब्राउझरमध्ये स्थान शोधणे समर्थित नाही.',
            'error-weather-failed': 'हवामानाचा डेटा मिळवण्यात अयशस्वी. कृपया पुन्हा प्रयत्न करा.',
            'result-title-weather': 'हवामानाचा तपशील',
            'result-title-crop': 'पिकाच्या शिफारसी',
            'result-title-fert': 'खत योजना',
            'result-title-explanation': 'स्पष्टीकरण',
            'result-metadata-time': 'अनुमान वेळ:',
            'result-metadata-version': 'मॉडेल आवृत्ती:',
            'result-recommendation-prefix': 'आम्ही लागवड करण्याची शिफारस करतो',
            'result-explanation-prefix': 'आमच्या विश्लेषणानुसार, तुमची माती प्रामुख्याने आहे',
            'result-explanation-mid': 'या प्रकारची माती शेतीसाठी उपयुक्त आहे',
            'result-explanation-suffix': 'शिफारस केलेले खत,',
            'result-explanation-end': 'निरोगी पिकांसाठी आवश्यक पोषक तत्वे पुरवण्याची शिफारस केली जाते।'
        }
    };
    
    function updateContent(lang) {
    
        document.getElementById('page-title').textContent = translations[lang]['page-title'];
        // The nav links need to be found by ID for translation
        const navHome = document.getElementById('nav-home');
        if(navHome) navHome.textContent = translations[lang]['nav-home'];
        document.getElementById('nav-predict').textContent = translations[lang]['nav-predict'];
        document.getElementById('nav-about').textContent = translations[lang]['nav-about'];
        document.getElementById('nav-support').textContent = translations[lang]['nav-support'];

        // Form elements
        document.getElementById('form-title').textContent = translations[lang]['form-title'];
        document.getElementById('form-subtitle').textContent = translations[lang]['form-subtitle'];
        document.getElementById('label-soil-image').textContent = translations[lang]['label-soil-image'];
        document.getElementById('label-location').textContent = translations[lang]['label-location'];
        document.getElementById('label-npk').textContent = translations[lang]['label-npk'];
        document.getElementById('get-location-btn').textContent = translations[lang]['btn-location'];
        
        // Check for manual button before translating
        const manualBtn = document.getElementById('get-manual-location-btn');
        if (manualBtn) {
            manualBtn.textContent = translations[lang]['btn-manual-weather'];
        }

        document.getElementById('submit-btn').textContent = translations[lang]['btn-submit'];
        document.getElementById('loader-text').textContent = translations[lang]['loader-text'];
    }

    const form = document.getElementById('prediction-form');
    const imageInput = document.getElementById('soil-image');
    const stateInput = document.getElementById('state-input');
    const regionInput = document.getElementById('region-input');
    const nitrogenInput = document.getElementById('nitrogen-input');
    const phosphorousInput = document.getElementById('phosphorous-input');
    const potassiumInput = document.getElementById('potassium-input');
    const getLocationBtn = document.getElementById('get-location-btn');
    const getManualLocationBtn = document.getElementById('get-manual-location-btn');
    const submitBtn = document.getElementById('submit-btn');
    const weatherDisplay = document.getElementById('weather-details-display');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');
    const resultsContainer = document.getElementById('prediction-results');
    const formContainer = document.getElementById('prediction-form-container');
    const languageSelect = document.getElementById('language-select');

    const API_PREDICT_URL = 'http://localhost:8000/api/predict/soil';
    const API_WEATHER_URL = 'http://localhost:8000/api/weather';
    const API_GEOCODE_URL = 'http://localhost:8000/api/geocode';

    let currentLat = null;
    let currentLng = null;

    // Event listener for language selector
    if (languageSelect) {
        languageSelect.addEventListener('change', (e) => {
            const selectedLang = e.target.value;
            updateContent(selectedLang);
        });
    }

    // Call to update content based on default language on load
    updateContent(languageSelect ? languageSelect.value : 'en'); 
    
    // Function to fetch and display weather details
    const fetchAndDisplayWeather = async (lat, lng) => {
        try {
            const weatherResponse = await fetch(`${API_WEATHER_URL}?lat=${lat}&lng=${lng}`);
            const weatherData = await weatherResponse.json();
            const lang = languageSelect.value;

            weatherDisplay.innerHTML = `
                <strong>${translations[lang]['weather-country-label']}:</strong> ${weatherData.country}<br>
                <strong>${translations[lang]['weather-state-label']}:</strong> ${weatherData.state}<br>
                <strong>${translations[lang]['weather-temp-label']}:</strong> ${weatherData.temperature.toFixed(2)} °C<br>
                <strong>${translations[lang]['weather-humidity-label']}:</strong> ${weatherData.humidity.toFixed(2)}%<br>
                <strong>${translations[lang]['weather-moisture-label']}:</strong> ${weatherData.moisture.toFixed(2)}%
            `;
            weatherDisplay.style.display = 'block';
            submitBtn.disabled = false;
            currentLat = lat;
            currentLng = lng;
        } catch (err) {
            showError(translations[languageSelect.value]['error-weather-failed']);
            submitBtn.disabled = true;
        }
    };

    // Event listener for "Use My Current Location" button
    getLocationBtn.addEventListener('click', () => {
        weatherDisplay.innerHTML = translations[languageSelect.value]['status-detecting'];
        weatherDisplay.style.display = 'block';
        submitBtn.disabled = true;

        if ("geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    // This is the call that gets the weather data
                    fetchAndDisplayWeather(position.coords.latitude, position.coords.longitude);
                },
                (err) => {
                    showError(translations[languageSelect.value]['error-geolocation-failed'] + ` ${err.message}`);
                    weatherDisplay.innerHTML = translations[languageSelect.value]['status-not-detected'];
                    submitBtn.disabled = true;
                }
            );
        } else {
            showError(translations[languageSelect.value]['error-geolocation-unsupported']);
            weatherDisplay.innerHTML = translations[languageSelect.value]['status-not-detected'];
            submitBtn.disabled = true;
        }
    });

    // Event listener for "Get Weather" button
    if (getManualLocationBtn) {
        getManualLocationBtn.addEventListener('click', async () => {
            const state = stateInput.value;
            const region = regionInput.value;

            if (!state || !region) {
                showError("Please enter both a state and a region.");
                return;
            }

            try {
                // First, geocode the text location to get coordinates
                const geocodeResponse = await fetch(`${API_GEOCODE_URL}?state=${state}&region=${region}`);
                const geocodeData = await geocodeResponse.json();
                
                if (geocodeData.latitude && geocodeData.longitude) {
                    // Second, use the coordinates to fetch and display the weather
                    fetchAndDisplayWeather(geocodeData.latitude, geocodeData.longitude);
                } else {
                    showError('Could not find coordinates for the entered location.');
                }
            } catch (err) {
                showError('Failed to get location coordinates. Please try again.');
            }
        });
    }

    // Handle form submission - FIX APPLIED HERE
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();
        showLoader();
        
        if (!currentLat || !currentLng) {
            showError('Please provide a location first.');
            hideLoader();
            return;
        }

        const formData = new FormData();
        // Check if an image file is selected before appending
        if (imageInput.files.length > 0) {
            formData.append('image', imageInput.files[0]);
        }
        
        formData.append('lat', currentLat);
        formData.append('lng', currentLng);
        
        // Append NPK values only if they are not empty
        if (nitrogenInput.value) formData.append('nitrogen', nitrogenInput.value);
        if (phosphorousInput.value) formData.append('phosphorous', phosphorousInput.value);
        if (potassiumInput.value) formData.append('potassium', potassiumInput.value);

        try {
            const response = await fetch(API_PREDICT_URL, {
                method: 'POST',
                body: formData,
            });

            let data;
            
            // If response status is bad (4xx, 5xx), attempt to get error detail
            if (!response.ok) {
                let errorMessageDetail = `Server error: ${response.status}`;
                try {
                    // Try to read JSON error body from server
                    const errorData = await response.json();
                    errorMessageDetail = errorData.detail || errorMessageDetail;
                } catch (jsonError) {
                    // Fallback if the bad status has no JSON body (e.g., 500 from proxy)
                    errorMessageDetail += " (Could not parse error details)";
                }
                throw new Error(errorMessageDetail);
            }
            
            // CRITICAL FIX: Robustly check response for valid JSON data
            const contentType = response.headers.get("content-type");
            if (!contentType || !contentType.includes("application/json")) {
                throw new Error("Prediction API returned a non-JSON or empty response.");
            }
            
            try {
                 data = await response.json();
            } catch (jsonParseError) {
                 throw new Error("Failed to parse prediction response as JSON.");
            }

            // Check if essential data is missing (e.g., if the model returned a success status 
            // but an empty payload, which should be caught by backend, but checked here for safety)
            if (!data || !data.crop_suggestions || data.crop_suggestions.length === 0) {
                 throw new Error("Prediction successful, but no valid results were returned.");
            }

            displayResults(data);

        } catch (error) {
            showError(`Prediction failed: ${error.message}`);
        } finally {
            hideLoader();
        }
    });

    // Helper functions
    function showLoader() {
        loader.style.display = 'flex';
        submitBtn.disabled = true;
        formContainer.style.display = 'none';
        resultsContainer.style.display = 'none'; // Ensure results are hidden when loading
    }

    function hideLoader() {
        loader.style.display = 'none';
        submitBtn.disabled = false;
        formContainer.style.display = 'block';
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        resultsContainer.style.display = 'none'; // Ensure results are hidden if error occurs
    }

    function hideError() {
        errorMessage.style.display = 'none';
    }

    function displayResults(data) {
        resultsContainer.innerHTML = '';
        resultsContainer.style.display = 'block';

        const weatherData = data.weather_data;
        const lang = languageSelect.value;
        const weatherHTML = `
            <div class="result-card">
                <h3 class="result-title">${translations[lang]['result-title-weather']}</h3>
                <p class="result-text">
                    <strong>${translations[lang]['weather-country-label']}:</strong> ${weatherData.country}<br>
                    <strong>${translations[lang]['weather-state-label']}:</strong> ${weatherData.state}<br>
                    <strong>${translations[lang]['weather-temp-label']}:</strong> ${weatherData.temperature.toFixed(2)} °C<br>
                    <strong>${translations[lang]['weather-humidity-label']}:</strong> ${weatherData.humidity.toFixed(2)}%<br>
                    <strong>${translations[lang]['weather-moisture-label']}:</strong> ${weatherData.moisture.toFixed(2)}%
                </p>
            </div>
        `;

        const resultHTML = `
            <div class="result-card">
                <h3 class="result-title">${translations[lang]['result-title-crop']}</h3>
                <p class="result-text">${translations[lang]['result-recommendation-prefix']} <strong>${data.crop_suggestions[0].name}</strong>.</p>
            </div>
            <div class="result-card">
                <h3 class="result-title">${translations[lang]['result-title-fert']}</h3>
                <p class="result-text">Based on your soil, we recommend using <strong>${data.fertilizer_recommendations[0].name}</strong>. Apply it as follows: <strong>${data.fertilizer_recommendations[0].amount}</strong>, with a frequency of <strong>${data.fertilizer_recommendations[0].frequency}</strong>.</p>
            </div>
            <div class="result-card">
                <h3 class="result-title">${translations[lang]['result-title-explanation']}</h3>
                <p class="result-text">${data.explanation}</p>
            </div>
            <p class="result-metadata">${translations[lang]['result-metadata-time']} ${data.inference_time_ms.toFixed(2)} ms | ${translations[lang]['result-metadata-version']} ${data.model_version}</p>
        `;

        resultsContainer.innerHTML = weatherHTML + resultHTML;
        formContainer.style.display = 'none';
    }
    
    // Initial content update based on default language
    updateContent(languageSelect ? languageSelect.value : 'en');
});