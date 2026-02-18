import streamlit as st
import numpy as np
import cv2
import joblib
import random
from skimage.feature import hog
from PIL import Image
import io

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Paws & Pixels 🐾",
    page_icon="🐾",
    layout="wide"
)

# --------------------------------------------------
# PROFESSIONAL CSS (FIXED & CLEANED)
# --------------------------------------------------
st.markdown("""
<style>

/* Animated Gradient Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Make normal text white */
[data-testid="stAppViewContainer"] {
    color: white;
}

/* Title */
.title-text {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(90deg,#ffd89b,#ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle-text {
    font-size: 1.2rem;
    opacity: 0.9;
}

/* File uploader clean style */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.18);
    padding: 18px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
}

section[data-testid="stFileUploader"] label {
    color: white !important;
    font-weight: 600;
}

section[data-testid="stFileUploader"] small {
    color: white !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg,#36d1dc,#5b86e5);
    color: white !important;
    border-radius: 15px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 20px rgba(0,0,0,0.4);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/svm_cat_dog_model.pkl")
    scaler = joblib.load("models/feature_scaler.pkl")
    return model, scaler

model, scaler = load_model()

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
IMAGE_SIZE = 128
CLASSES = ["Cat 🐱", "Dog 🐶"]

CAT_FACTS = [
    "Cats sleep for nearly 70% of their lives 😴",
    "Each cat's nose is unique like fingerprints 🐾",
    "Cats can rotate their ears 180 degrees 🎧",
    "Ancient Egyptians worshipped cats 👑",
    "Cats can make over 100 vocal sounds 🎵",
    "Cats sleep for 12–16 hours a day 😴",
    "A cat’s nose print is unique, just like a human fingerprint 🐾",
    "Cats can rotate their ears 180 degrees 🎧",
    "They can jump up to 6 times their body length 🏆",
    "Cats have five toes on front paws but only four on the back 🐾",
    "Ancient Egyptians considered cats sacred animals 👑",
    "Cats can make over 100 different sounds 🎵"
    ]

DOG_FACTS = [
    "Dogs can understand up to 250 words 🧠",    
    "Dogs can smell your emotions ❤️",
    "Some dogs can detect diseases 🏥",
    "Dogs dream just like humans 💤",
    "Dogs can understand up to 250 words and gestures 🧠",
    "A dog’s nose print is unique just like a fingerprint 🐶",
    "Dogs can smell up to 100,000 times better than humans 👃",
    "Puppies are born deaf and blind 👶",
    "Dogs can detect certain diseases like cancer 🏥",
    "Dogs dream just like humans 💤",
    "Some dogs can learn over 1,000 words 📚",
    "A dog’s sense of smell is so strong it can detect emotions ❤️",
    "Dogs have three eyelids 👀",
    "The Basenji dog doesn’t bark, it yodels 🎶"

]

# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image

def extract_hog_features(image):
    return hog(
        image,
        orientations=6,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

# --------------------------------------------------
# SESSION NAVIGATION
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ==================================================
# HOME PAGE
# ==================================================
# ==================================================
# HOME PAGE
# ==================================================
if st.session_state.page == "home":

    col1, col2 = st.columns([1.4, 1])   # 👈 updated ratio

    with col1:
        st.markdown("<div class='title-text'>Paws & Pixels 🐾</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle-text'>A professional SVM classifier that intelligently distinguishes between Cats and Dogs.</div>", unsafe_allow_html=True)
        st.markdown("### Upload a cat or dog image and let machine do the magic ✨")

        uploaded_file = st.file_uploader("Drop your image here 👇", type=["jpg","jpeg","png"])

        if uploaded_file:
            st.session_state.image_bytes = uploaded_file.read()
            st.session_state.page = "result"
            st.rerun()

    with col2:
        st.image(
        "assets/cat_dog.jpg",
        use_container_width=True)




# ==================================================
# RESULT PAGE
# ==================================================
elif st.session_state.page == "result":

    image_bytes = st.session_state.image_bytes
    image = Image.open(io.BytesIO(image_bytes))

    processed = preprocess_image(image_bytes)
    features = extract_hog_features(processed)
    features = scaler.transform([features])

    prediction = model.predict(features)[0]
    label = CLASSES[prediction]
    fact_list = CAT_FACTS if prediction == 0 else DOG_FACTS
    fact = random.choice(fact_list)

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown(f"# 🎯 Prediction: {label}")

        st.markdown("### 💡 Want a fun fact?")
        if st.button("Reveal Fun Fact 🎉"):
            st.success(fact)

        if st.button("🔄 Try Another Image"):
            st.session_state.page = "home"
            st.rerun()
