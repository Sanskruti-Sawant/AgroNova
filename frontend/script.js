
document.addEventListener('DOMContentLoaded', () => {
    
    const translations = {
        'en': {
            'nav-home': 'Home',
            'nav-about': 'About us',
            'nav-support': 'Support',
            'btn-get-started': 'GET STARTED',
            'hero-title': 'Your Personal Farming Assistant',
            'hero-subtitle': 'Use the power of AI to get personalized crop and fertilizer recommendations. Simply upload a soil image and your location.',
            'reviews-title': 'What Our Users Say',
            'review-quote-1': '“This app is a game-changer! My farm has never been more productive.”',
            'review-author-1': 'Amit K.',
            'review-title-1': 'Farmer',
            'review-quote-2': '“The fertilizer recommendations are spot on. I got my crop recommendations in just a few seconds!”',
            'review-author-2': 'Pooja S.',
            'review-title-2': 'Agricultural Student',
            'review-quote-3': '“I got my crop recommendations in just a few seconds. This app is a game-changer for my farm!”',
            'review-author-3': 'Rakesh M.',
            'review-title-3': 'Agri-entrepreneur'
        },
        'hi': {
            'nav-home': 'होम',
            'nav-about': 'हमारे बारे में',
            'nav-support': 'सहयोग',
            'btn-get-started': 'शुरू करें',
            'hero-title': 'आपका व्यक्तिगत खेती सहायक',
            'hero-subtitle': 'एआई की शक्ति का उपयोग करके व्यक्तिगत फसल और खाद की सिफारिशें प्राप्त करें। बस मिट्टी की तस्वीर और अपना स्थान अपलोड करें।',
            'reviews-title': 'हमारे उपयोगकर्ता क्या कहते हैं',
            'review-quote-1': '“यह ऐप एक गेम-चेंजर है! मेरा खेत इतना उत्पादक कभी नहीं रहा।”',
            'review-author-1': 'अमित के.',
            'review-title-1': 'किसान',
            'review-quote-2': '“खाद की सिफारिशें सटीक हैं। मुझे कुछ ही सेकंड में अपनी फसल की सिफारिशें मिल गईं!”',
            'review-author-2': 'पूजा एस.',
            'review-title-2': 'कृषि छात्र',
            'review-quote-3': '“मुझे कुछ ही सेकंड में अपनी फसल की सिफारिशें मिल गईं। यह ऐप मेरे खेत के लिए एक गेम-चेंजर है!”',
            'review-author-3': 'राकेश एम.',
            'review-title-3': 'कृषि उद्यमी'
        },
        'mr': {
            'nav-home': 'मुख्यपृष्ठ',
            'nav-about': 'आमच्याबद्दल',
            'nav-support': 'सहाय्य',
            'btn-get-started': 'सुरू करा',
            'hero-title': 'तुमचा वैयक्तिक शेती सहायक',
            'hero-subtitle': 'तुमच्या मातीचा फोटो आणि स्थानाचा वापर करून AI च्या मदतीने वैयक्तिक पिकांच्या आणि खतांच्या शिफारसी मिळवा.',
            'reviews-title': 'आमचे वापरकर्ते काय म्हणतात',
            'review-quote-1': '“हे ॲप एक गेम-चेंजर आहे! माझ्या शेतात यापूर्वी इतके उत्पादन कधीच झाले नाही.”',
            'review-author-1': 'अमित के.',
            'review-title-1': 'शेतकरी',
            'review-quote-2': '“खतांच्या शिफारसी अगदी अचूक आहेत. मला काही सेकंदातच माझ्या पिकांच्या शिफारसी मिळाल्या!”',
            'review-author-2': 'पूजा एस.',
            'review-title-2': 'कृषि विद्यार्थी',
            'review-quote-3': '“मला काही सेकंदातच माझ्या पिकांच्या शिफारसी मिळाल्या. हे ॲप माझ्या शेतीसाठी एक गेम-चेंजर आहे!”',
            'review-author-3': 'राकेश एम.',
            'review-title-3': 'कृषि उद्यमी'
        }
    };
    
    // Function to update all text on the page
    function updateContent(lang) {
        document.getElementById('nav-home').textContent = translations[lang]['nav-home'];
        document.getElementById('nav-about').textContent = translations[lang]['nav-about'];
        document.getElementById('nav-support').textContent = translations[lang]['nav-support'];
        document.querySelector('.get-started-btn').textContent = translations[lang]['btn-get-started'];
        document.getElementById('hero-title').textContent = translations[lang]['hero-title'];
        document.getElementById('hero-subtitle').textContent = translations[lang]['hero-subtitle'];
        document.getElementById('reviews-title').textContent = translations[lang]['reviews-title'];
        
        // Update review card texts
        const reviewCards = document.querySelectorAll('.review-card');
        if (reviewCards.length > 0) {
            document.getElementById('review-quote-1').textContent = translations[lang]['review-quote-1'];
            document.getElementById('review-author-1').textContent = translations[lang]['review-author-1'];
            document.getElementById('review-title-1').textContent = translations[lang]['review-title-1'];
            
            document.getElementById('review-quote-2').textContent = translations[lang]['review-quote-2'];
            document.getElementById('review-author-2').textContent = translations[lang]['review-author-2'];
            document.getElementById('review-title-2').textContent = translations[lang]['review-title-2'];
            
            document.getElementById('review-quote-3').textContent = translations[lang]['review-quote-3'];
            document.getElementById('review-author-3').textContent = translations[lang]['review-author-3'];
            document.getElementById('review-title-3').textContent = translations[lang]['review-title-3'];
        }
    }

    // Event listener for the language selector
    const languageSelect = document.getElementById('language-select');
    if (languageSelect) {
        languageSelect.addEventListener('change', (e) => {
            const selectedLang = e.target.value;
            updateContent(selectedLang);
        });
    }

    // Your existing carousel code
    const carousel = document.querySelector('.review-carousel');
    const cards = document.querySelectorAll('.review-card');
    const dotsContainer = document.querySelector('.carousel-dots-container');
    const dots = document.querySelectorAll('.dot');
    const prevBtn = document.getElementById('prev-review-btn');
    const nextBtn = document.getElementById('next-review-btn');
    const totalCards = cards.length;
    let currentIndex = 0;
    let autoSlideInterval;
    
    function updateCarousel() {
        if (!carousel || !dots) return;
        const offset = -currentIndex * 100;
        carousel.style.transform = `translateX(${offset}%)`;
        dots.forEach((dot, index) => {
            dot.classList.toggle('active', index === currentIndex);
        });
    }

    function nextReview() {
        currentIndex = (currentIndex + 1) % totalCards;
        updateCarousel();
    }
    
    function prevReview() {
        currentIndex = (currentIndex - 1 + totalCards) % totalCards;
        updateCarousel();
    }
    
    function startAutoSlide() {
        autoSlideInterval = setInterval(nextReview, 5000);
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            clearInterval(autoSlideInterval);
            nextReview();
            startAutoSlide();
        });
    }
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            clearInterval(autoSlideInterval);
            prevReview();
            startAutoSlide();
        });
    }
    
    if (dotsContainer) {
        dots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                clearInterval(autoSlideInterval);
                currentIndex = index;
                updateCarousel();
                startAutoSlide();
            });
        });
    }

    const getStartedBtn = document.querySelector('.get-started-btn-large');
    if (getStartedBtn) {
        getStartedBtn.addEventListener('click', () => {
            window.location.href = 'predict.html';
        });
    }
    
    if (carousel) {
        updateCarousel();
        startAutoSlide();
    }
   
    updateContent('en');
});