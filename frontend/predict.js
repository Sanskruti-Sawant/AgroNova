// predict.js — fixed & hardened
document.addEventListener("DOMContentLoaded", () => {
  // ---------------------------
  // i18n
  // ---------------------------
  const translations = {
    en: {
      "page-title": "AgriNova | Predict",
      "nav-home": "Home",
      "nav-predict": "Predict",
      "nav-about": "About us",
      "nav-support": "Support",
      "form-title": "Start Your Prediction",
      "form-subtitle":
        "Upload a soil image and provide your location for a personalized plan.",
      "label-soil-image": "Soil Image (JPG/PNG)",
      "label-location": "Your Location",
      "label-npk": "Optional NPK Values",
      "btn-location": "Use My Current Location",
      "btn-manual-weather": "Get Weather",
      "btn-submit": "Get Prediction",
      "loader-text": "Analyzing soil... this may take a moment.",
      "status-detecting": "Detecting your location...",
      "status-not-detected": "Location not detected.",
      "error-geolocation-failed": "Geolocation failed:",
      "error-geolocation-unsupported":
        "Geolocation is not supported by this browser.",
      "error-weather-failed": "Failed to fetch weather data. Please try again.",
      "result-title-weather": "Weather Details",
      "result-title-crop": "Crop Suggestions",
      "result-title-fert": "Fertilizer Plan",
      "result-title-explanation": "Explanation",
      "result-metadata-time": "Inference Time:",
      "result-metadata-version": "Model Version:",
      "result-recommendation-prefix": "We recommend planting",
      "result-explanation-prefix":
        "Based on our analysis, your soil is primarily",
      "result-explanation-mid":
        "This type of soil is well-suited for growing",
      "result-explanation-suffix": "The recommended fertilizer,",
      "result-explanation-end":
        "is suggested to provide the necessary nutrients for a healthy crop yield.",
      "weather-temp-label": "Temperature",
      "weather-humidity-label": "Humidity",
      "weather-moisture-label": "Moisture",
      "weather-country-label": "Country",
      "weather-state-label": "State",
      "result-top3": "Top 3 predictions",
      // small UI messages
      "need-state-region": "Please enter both a state and a region.",
      "cant-find-coords": "Could not find coordinates for the entered location.",
      "need-location-first": "Please provide a location first.",
      "prediction-empty":
        "Prediction successful, but no valid results were returned.",
    },
    hi: {
      "page-title": "कृषि-नोवा | अनुमान",
      "nav-home": "होम",
      "nav-predict": "अनुमान",
      "nav-about": "हमारे बारे में",
      "nav-support": "सहयोग",
      "form-title": "अपना अनुमान शुरू करें",
      "form-subtitle": "मिट्टी की तस्वीर अपलोड करें और अपनी जगह की जानकारी दें।",
      "label-soil-image": "मिट्टी की तस्वीर (JPG/PNG)",
      "label-location": "आपकी जगह",
      "label-npk": "वैकल्पिक NPK मान",
      "btn-location": "मेरी वर्तमान जगह का उपयोग करें",
      "btn-manual-weather": "मौसम प्राप्त करें",
      "btn-submit": "अनुमान प्राप्त करें",
      "loader-text":
        "मिट्टी का विश्लेषण हो रहा है... इसमें थोड़ा समय लग सकता है।",
      "status-detecting": "आपकी जगह का पता लगाया जा रहा है...",
      "status-not-detected": "जगह का पता नहीं चला।",
      "error-geolocation-failed": "जगह की जानकारी विफल:",
      "error-geolocation-unsupported":
        "आपके ब्राउज़र में जगह की जानकारी समर्थित नहीं है।",
      "error-weather-failed":
        "मौसम का डेटा लाने में विफल। कृपया फिर से प्रयास करें।",
      "result-title-weather": "मौसम का विवरण",
      "result-title-crop": "फसल के सुझाव",
      "result-title-fert": "खाद योजना",
      "result-title-explanation": "स्पष्टीकरण",
      "result-metadata-time": "अनुमान समय:",
      "result-metadata-version": "मॉडल संस्करण:",
      "result-recommendation-prefix": "हम लगाने की सलाह देते हैं",
      "result-explanation-prefix":
        "हमारे विश्लेषण के आधार पर, आपकी मिट्टी मुख्य रूप से है",
      "result-explanation-mid": "इस प्रकार की मिट्टी में खेती के लिए उपयुक्त है",
      "result-explanation-suffix": "अनुशंसित उर्वरक,",
      "result-explanation-end":
        "को स्वस्थ फसल के लिए आवश्यक पोषक तत्व प्रदान करने का सुझाव दिया गया है।",
      "weather-temp-label": "तापमान",
      "weather-humidity-label": "आर्द्रता",
      "weather-moisture-label": "मातीतील ओलावा",
      "weather-country-label": "देश",
      "weather-state-label": "राज्य",
      "result-top3": "शीर्ष 3 भविष्यवाणियाँ",
      "need-state-region": "कृपया राज्य और क्षेत्र दोनों दर्ज करें।",
      "cant-find-coords": "दिए गए स्थान के निर्देशांक नहीं मिल सके।",
      "need-location-first": "कृपया पहले स्थान प्रदान करें।",
      "prediction-empty":
        "पूर्वानुमान सफल रहा, लेकिन कोई मान्य परिणाम नहीं मिला।",
    },
    mr: {
      "page-title": "ॲग्रीनोव्हा | अनुमान",
      "nav-home": "मुख्यपृष्ठ",
      "nav-predict": "अनुमान",
      "nav-about": "आमच्याबद्दल",
      "nav-support": "सहाय्य",
      "form-title": "तुमचा अनुमान सुरू करा",
      "form-subtitle":
        "मातीचा फोटो अपलोड करा आणि तुमच्या स्थानाची माहिती द्या.",
      "label-soil-image": "मातीचा फोटो (JPG/PNG)",
      "label-location": "तुमचे स्थान",
      "label-npk": "वैकल्पिक NPK मूल्ये",
      "btn-location": "माझ्या वर्तमान स्थानाचा वापर करा",
      "btn-manual-weather": "हवामान मिळवा",
      "btn-submit": "अनुमान मिळवा",
      "loader-text":
        "मातीचे विश्लेषण होत आहे... यास थोडा वेळ लागू शकतो.",
      "status-detecting": "तुमचे स्थान शोधले जात आहे...",
      "status-not-detecte d": "स्थान सापडले नाही.",
      "error-geolocation-failed": "स्थान शोधण्यात अयशस्वी:",
      "error-geolocation-unsupported":
        "तुमच्या ब्राउझरमध्ये स्थान शोधणे समर्थित नाही.",
      "error-weather-failed":
        "हवामानाचा डेटा मिळवण्यात अयशस्वी. कृपया पुन्हा प्रयत्न करा.",
      "result-title-weather": "हवामानाचा तपशील",
      "result-title-crop": "पिकाच्या शिफारसी",
      "result-title-fert": "खत योजना",
      "result-title-explanation": "स्पष्टीकरण",
      "result-metadata-time": "अनुमान वेळ:",
      "result-metadata-version": "मॉडेल आवृत्ती:",
      "result-recommendation-prefix": "आम्ही लागवड करण्याची शिफारस करतो",
      "result-explanation-prefix":
        "आमच्या विश्लेषणानुसार, तुमची माती प्रामुख्याने आहे",
      "result-explanation-mid": "या प्रकारची माती शेतीसाठी उपयुक्त आहे",
      "result-explanation-suffix": "शिफारस केलेले खत,",
      "result-explanation-end":
        "निरोगी पिकांसाठी आवश्यक पोषक तत्वे पुरवण्याची शिफारस केली जाते。",
      "result-top3": "टॉप 3 अंदाज",
      "need-state-region": "कृपया राज्य आणि प्रदेश दोन्ही भरा.",
      "cant-find-coords": "दिलेल्या ठिकाणाचे निर्देशांक सापडले नाहीत.",
      "need-location-first": "कृपया आधी स्थान द्या.",
      "prediction-empty":
        "भविष्यवाणी यशस्वी, परंतु कोणतेही वैध परिणाम नाहीत.",
    },
  };

  const languageSelect = document.getElementById("language-select");
  const t = (key, lang) => {
    const l = lang || (languageSelect ? languageSelect.value : "en");
    return (translations[l] && translations[l][key]) ||
      (translations["en"] && translations["en"][key]) ||
      key;
  };

  function updateContent(lang) {
    const setText = (id, key) => {
      const el = document.getElementById(id);
      if (el) el.textContent = t(key, lang);
    };
    setText("page-title", "page-title");
    setText("nav-home", "nav-home");
    setText("nav-predict", "nav-predict");
    setText("nav-about", "nav-about");
    setText("nav-support", "nav-support");
    setText("form-title", "form-title");
    setText("form-subtitle", "form-subtitle");
    setText("label-soil-image", "label-soil-image");
    setText("label-location", "label-location");
    setText("label-npk", "label-npk");
    setText("get-location-btn", "btn-location");
    setText("get-manual-location-btn", "btn-manual-weather");
    setText("submit-btn", "btn-submit");
    setText("loader-text", "loader-text");
  }

  if (languageSelect) {
    languageSelect.addEventListener("change", (e) =>
      updateContent(e.target.value)
    );
  }

  // ---------------------------
  // DOM refs
  // ---------------------------
  const form = document.getElementById("prediction-form");
  const imageInput = document.getElementById("soil-image");
  const stateInput = document.getElementById("state-input");
  const regionInput = document.getElementById("region-input");
  const nitrogenInput = document.getElementById("nitrogen-input");
  const phosphorousInput = document.getElementById("phosphorous-input");
  const potassiumInput = document.getElementById("potassium-input");
  const getLocationBtn = document.getElementById("get-location-btn");
  const getManualLocationBtn = document.getElementById(
    "get-manual-location-btn"
  );
  const submitBtn = document.getElementById("submit-btn");
  const weatherDisplay = document.getElementById("weather-details-display");
  const loader = document.getElementById("loader");
  const errorMessage = document.getElementById("error-message");
  const resultsContainer = document.getElementById("prediction-results");
  const formContainer = document.getElementById("prediction-form-container");
  const apiBadge = document.getElementById("api-status-badge");

  // ---------------------------
  // API base & endpoints
  // ---------------------------
  function getApiBase() {
    const el = document.body;
    const attr = el ? el.getAttribute("data-api-base") : null;
    // default to backend container
    return (attr && attr.trim()) || "http://localhost:8000";
  }
  const API_BASE = getApiBase();
  const API_PREDICT_URL = `${API_BASE}/api/predict/soil`;
  const API_WEATHER_URL = `${API_BASE}/api/weather`;
  const API_GEOCODE_URL = `${API_BASE}/api/geocode`;
  const API_HEALTH_URL = `${API_BASE}/health`;

  // ---------------------------
  // Health check badge
  // ---------------------------
  async function pingHealth() {
    try {
      const res = await fetch(API_HEALTH_URL, { method: "GET" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (apiBadge) {
        apiBadge.textContent = `API: Connected (${data.message || "OK"})`;
        apiBadge.classList.remove("bad");
        apiBadge.classList.add("ok");
      }
    } catch (e) {
      if (apiBadge) {
        apiBadge.textContent = `API: Offline (${e.message})`;
        apiBadge.classList.remove("ok");
        apiBadge.classList.add("bad");
      }
    }
  }
  pingHealth();

  // ---------------------------
  // UI helpers
  // ---------------------------
  function showLoader() {
    if (loader) loader.style.display = "flex";
    if (submitBtn) submitBtn.disabled = true;
    if (formContainer) formContainer.style.display = "none";
    if (resultsContainer) resultsContainer.style.display = "none";
  }
  function hideLoader() {
    if (loader) loader.style.display = "none";
    if (submitBtn) submitBtn.disabled = false;
    if (formContainer) formContainer.style.display = "block";
  }
  function showError(message) {
    if (errorMessage) {
      errorMessage.textContent = message;
      errorMessage.style.display = "block";
    }
    if (resultsContainer) resultsContainer.style.display = "none";
  }
  function hideError() {
    if (errorMessage) errorMessage.style.display = "none";
  }
  function ensureJsonResponseOk(res, endpointName) {
    if (!res.ok) {
      throw new Error(`${endpointName} failed with HTTP ${res.status}`);
    }
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      throw new Error(`${endpointName} returned non-JSON response`);
    }
  }

  // ---------------------------
  // Weather fetch + display
  // ---------------------------
  let currentLat = null;
  let currentLng = null;

  async function fetchAndDisplayWeather(lat, lng) {
    try {
      const url = `${API_WEATHER_URL}?lat=${encodeURIComponent(
        lat
      )}&lng=${encodeURIComponent(lng)}`;
      const weatherResponse = await fetch(url, { method: "GET" });
      ensureJsonResponseOk(weatherResponse, "Weather");
      const weatherData = await weatherResponse.json();

      const lang = languageSelect ? languageSelect.value : "en";
      if (weatherDisplay) {
        weatherDisplay.innerHTML = `
          <strong>${t("weather-country-label", lang)}:</strong> ${
          weatherData.country
        }<br>
          <strong>${t("weather-state-label", lang)}:</strong> ${
          weatherData.state
        }<br>
          <strong>${t("weather-temp-label", lang)}:</strong> ${Number(
          weatherData.temperature
        ).toFixed(2)} °C<br>
          <strong>${t("weather-humidity-label", lang)}:</strong> ${Number(
          weatherData.humidity
        ).toFixed(2)}%<br>
          <strong>${t("weather-moisture-label", lang)}:</strong> ${Number(
          weatherData.moisture
        ).toFixed(2)}%
        `;
        weatherDisplay.style.display = "block";
      }
      if (submitBtn) submitBtn.disabled = false;

      currentLat = lat;
      currentLng = lng;
    } catch (err) {
      console.error("Weather fetch error:", err);
      showError(`${t("error-weather-failed")} (${err.message})`);
      if (submitBtn) submitBtn.disabled = true;
      if (weatherDisplay) weatherDisplay.style.display = "none";
      currentLat = null;
      currentLng = null;
    }
  }

  // ---------------------------
  // Location buttons
  // ---------------------------
  if (getLocationBtn) {
    getLocationBtn.addEventListener("click", () => {
      hideError();
      if (weatherDisplay) {
        weatherDisplay.textContent = t("status-detecting");
        weatherDisplay.style.display = "block";
      }
      if (submitBtn) submitBtn.disabled = true;

      if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            fetchAndDisplayWeather(
              position.coords.latitude,
              position.coords.longitude
            );
          },
          (err) => {
            console.error("Geolocation error:", err);
            showError(`${t("error-geolocation-failed")} ${err.message}`);
            if (weatherDisplay)
              weatherDisplay.textContent = t("status-not-detected");
            if (submitBtn) submitBtn.disabled = true;
          },
          { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
        );
      } else {
        showError(t("error-geolocation-unsupported"));
        if (weatherDisplay)
          weatherDisplay.textContent = t("status-not-detected");
        if (submitBtn) submitBtn.disabled = true;
      }
    });
  }

  if (getManualLocationBtn) {
    getManualLocationBtn.addEventListener("click", async () => {
      hideError();
      const state = (stateInput && stateInput.value ? stateInput.value : "")
        .trim();
      const region = (regionInput && regionInput.value
        ? regionInput.value
        : ""
      ).trim();

      if (!state || !region) {
        showError(t("need-state-region"));
        return;
      }

      try {
        const url = `${API_GEOCODE_URL}?state=${encodeURIComponent(
          state
        )}&region=${encodeURIComponent(region)}`;
        const geocodeResponse = await fetch(url);
        ensureJsonResponseOk(geocodeResponse, "Geocode");
        const geocodeData = await geocodeResponse.json();
        if (geocodeData.latitude && geocodeData.longitude) {
          fetchAndDisplayWeather(geocodeData.latitude, geocodeData.longitude);
        } else {
          showError(t("cant-find-coords"));
        }
      } catch (err) {
        console.error("Geocode error:", err);
        showError(`Failed to get location coordinates. (${err.message})`);
      }
    });
  }

  // ---------------------------
  // Submit prediction
  // ---------------------------
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError();
      showLoader();

      if (!currentLat || !currentLng) {
        showError(t("need-location-first"));
        hideLoader();
        return;
      }

      try {
        const formData = new FormData();
        if (imageInput && imageInput.files.length > 0) {
          formData.append("image", imageInput.files[0]);
        }
        formData.append("lat", String(currentLat));
        formData.append("lng", String(currentLng));
        if (nitrogenInput && nitrogenInput.value)
          formData.append("nitrogen", nitrogenInput.value);
        if (phosphorousInput && phosphorousInput.value)
          formData.append("phosphorous", phosphorousInput.value);
        if (potassiumInput && potassiumInput.value)
          formData.append("potassium", potassiumInput.value);

        const response = await fetch(API_PREDICT_URL, {
          method: "POST",
          body: formData,
        });
        ensureJsonResponseOk(response, "Prediction");
        const data = await response.json();

        if (
          !data ||
          !Array.isArray(data.crop_suggestions) ||
          data.crop_suggestions.length === 0
        ) {
          throw new Error(t("prediction-empty"));
        }
        displayResults(data);
      } catch (error) {
        console.error("Predict error:", error);
        showError(`Prediction failed: ${error.message}`);
      } finally {
        hideLoader();
      }
    });
  }

  // ---------------------------
  // Top-3 crop UI & results render
  // ---------------------------
  function renderTop3(cropSuggestions, lang) {
    const list = cropSuggestions.slice(0, 3);
    const safe = (v) => Math.max(0, Math.min(1, Number(v || 0)));
    const maxProb = Math.max(...list.map((c) => safe(c.confidence)));
    const items = list
      .map((c, idx) => {
        const pct = (safe(c.confidence) * 100).toFixed(1);
        const rel = maxProb > 0 ? Math.min(100, (safe(c.confidence) / maxProb) * 100) : 0;
        const name = c.name ?? c.crop ?? "—";
        return `
          <li class="top3-item">
            <div class="top3-row">
              <span class="rank">#${idx + 1}</span>
              <span class="crop-name">${name}</span>
              <span class="prob">${pct}%</span>
            </div>
            <div class="prob-bar">
              <div class="prob-bar-fill" style="width:${rel}%"></div>
            </div>
          </li>`;
      })
      .join("");

    return `
      <div class="top3-wrapper">
        <div class="top3-title">${t("result-top3", lang)}</div>
        <ul class="top3-list">${items}</ul>
      </div>`;
  }

  function displayResults(data) {
    if (!resultsContainer) return;
    resultsContainer.innerHTML = "";
    resultsContainer.style.display = "block";

    const lang = languageSelect ? languageSelect.value : "en";
    const weatherData = data.weather_data || {};

    const weatherHTML = `
      <div class="result-card">
        <h3 class="result-title">${t("result-title-weather", lang)}</h3>
        <p class="result-text">
          <strong>${t("weather-country-label", lang)}:</strong> ${weatherData.country ?? "—"}<br>
          <strong>${t("weather-state-label", lang)}:</strong> ${weatherData.state ?? "—"}<br>
          <strong>${t("weather-temp-label", lang)}:</strong> ${Number(weatherData.temperature || 0).toFixed(2)} °C<br>
          <strong>${t("weather-humidity-label", lang)}:</strong> ${Number(weatherData.humidity || 0).toFixed(2)}%<br>
          <strong>${t("weather-moisture-label", lang)}:</strong> ${Number(weatherData.moisture || 0).toFixed(2)}%
        </p>
      </div>`;

    const primary = data.crop_suggestions[0];
    const primaryName = primary?.name ?? primary?.crop ?? "—";
    const primaryLine = `
      <p class="result-text">
        ${t("result-recommendation-prefix", lang)} <strong>${primaryName}</strong>.
      </p>`;

    const top3HTML = renderTop3(data.crop_suggestions, lang);

    const fert =
      data.fertilizer_recommendations && data.fertilizer_recommendations[0]
        ? data.fertilizer_recommendations[0]
        : { name: "N/A", amount: "—", frequency: "—" };

    const fertHTML = `
      <div class="result-card">
        <h3 class="result-title">${t("result-title-fert", lang)}</h3>
        <p class="result-text">
          Based on your soil, we recommend using <strong>${fert.name}</strong>.
          Apply it as follows: <strong>${fert.amount}</strong>, with a frequency of
          <strong>${fert.frequency}</strong>.
        </p>
      </div>`;

    const explainHTML = `
      <div class="result-card">
        <h3 class="result-title">${t("result-title-explanation", lang)}</h3>
        <p class="result-text">${data.explanation || ""}</p>
      </div>`;

    const metaHTML = `
      <p class="result-metadata">
        ${t("result-metadata-time", lang)} ${Number(data.inference_time_ms || 0).toFixed(2)} ms |
        ${t("result-metadata-version", lang)} ${data.model_version || ""}
      </p>`;

    const cropHTML = `
      <div class="result-card">
        <h3 class="result-title">${t("result-title-crop", lang)}</h3>
        ${primaryLine}
        ${top3HTML}
      </div>`;

    resultsContainer.innerHTML =
      weatherHTML + cropHTML + fertHTML + explainHTML + metaHTML;

    if (formContainer) formContainer.style.display = "none";
  }

  // Initial content sync
  updateContent(languageSelect ? languageSelect.value : "en");
});
