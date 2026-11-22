"""
EcoSort AI - Intelligent Waste Classification Web Application
Powered by TensorFlow / Keras & Streamlit
"""
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow import keras
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort AI | Intelligent Waste Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, modern, nature/green-themed UI
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.2);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        margin-top: 0.6rem;
        opacity: 0.92;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Cards */
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #e0e7e1;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .custom-card:hover {
        box-shadow: 0 6px 18px rgba(46, 125, 50, 0.08);
    }

    /* Prediction Result Card */
    .result-card {
        background: linear-gradient(145deg, #ffffff 0%, #f4fbf5 100%);
        border: 2px solid #81c784;
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.12);
    }

    .badge-category {
        display: inline-block;
        padding: 0.35rem 1.1rem;
        font-size: 1.35rem;
        font-weight: 700;
        border-radius: 999px;
        margin-bottom: 0.5rem;
    }

    .badge-confidence {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.9rem;
        background-color: #e8f5e9;
        color: #1b5e20;
        font-weight: 700;
        border-radius: 999px;
        font-size: 0.95rem;
        border: 1px solid #a5d6a7;
    }

    /* Disposal Guideline Boxes */
    .bin-badge {
        display: inline-block;
        font-weight: 700;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        color: #ffffff;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }

    .do-box {
        background-color: #f1f8e9;
        color: #1B4332;
        border-left: 4px solid #4caf50;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.6rem;
    }
    .dont-box {
        background-color: #ffebee;
        color: #7F1D1D;
        border-left: 4px solid #ef5350;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.6rem;
    }

    .eco-tip-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 1px solid #a5d6a7;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: #1b5e20;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Probability Bar Custom Styling */
    .prob-bar-container {
        margin-bottom: 0.75rem;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .prob-bar-bg {
        width: 100%;
        height: 10px;
        background-color: #edf2ee;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }

    /* Upload Box Styling */
    .upload-hint {
        color: #556b2f;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Configuration & Waste Class Specifications
# -----------------------------------------------------------------------------
MODEL_FILE = "Improved_Waste_Classification.keras"
TARGET_SIZE = (224, 224)

# Strictly required class order: Glass, Metal, Organic, Paper, Plastic
CLASSES = ["Glass", "Metal", "Organic", "Paper", "Plastic"]

# Detailed disposal guidance, bin colors, dos, donts, and ecological impact
WASTE_DETAILS = {
    "Glass": {
        "icon": "🍾",
        "tagline": "100% Endlessly Recyclable",
        "color": "#1976D2",
        "badge_bg": "#E3F2FD",
        "bin_name": "Blue / Glass Recycling Bin",
        "bin_color": "#1976D2",
        "steps": [
            "Empty any liquid or food contents completely.",
            "Rinse clean with water to prevent mold and pest attraction.",
            "Remove metal lids or plastic caps (sort them into Metal / Plastic).",
            "Keep bottles and jars intact; do not deliberately crush or break."
        ],
        "allowed": [
            "Beverage bottles (soda, wine, beer, juice, water)",
            "Food containers & condiment jars (jam, sauce, pickles, mayo)",
            "Clean cosmetic and perfume bottles",
            "Clear, green, and brown glass jars"
        ],
        "prohibited": [
            "Window panes and automotive windshield glass",
            "Mirrors, picture frame glass, and eyeglasses",
            "Ceramics, porcelain tableware, Pyrex, or ovenware",
            "Light bulbs, fluorescent tubes, or lab glassware"
        ],
        "eco_fact": "Glass can be recycled forever with zero loss in quality or purity. Recycling just one glass bottle saves enough electricity to power a computer for 25 minutes!"
    },
    "Metal": {
        "icon": "🥫",
        "tagline": "High-Value Infinitely Recyclable Resource",
        "color": "#E65100",
        "badge_bg": "#FFF3E0",
        "bin_name": "Yellow / Metal Recyclables Bin",
        "bin_color": "#E65100",
        "steps": [
            "Empty completely and give a quick rinse to eliminate organic residues.",
            "Crush aluminum cans if convenient to conserve bin capacity.",
            "Clean foil trays and fold them into a ball (at least golf-ball size).",
            "Place loose metal caps inside a clean steel can and crimp slightly."
        ],
        "allowed": [
            "Aluminum drink cans (soda, iced tea, sparkling water)",
            "Steel and tin food cans (beans, soup, canned tuna, pet food)",
            "Clean aluminum foil and baking tins",
            "Metal bottle crowns, lids, and screw-caps"
        ],
        "prohibited": [
            "Pressurized aerosol cans containing hazardous contents",
            "Electronic waste, wires, and circuit components",
            "Household or automotive batteries (chemical hazard)",
            "Pots with non-stick Teflon coating or heavy motor components"
        ],
        "eco_fact": "Recycling aluminum requires 95% less energy than refining raw bauxite ore. A recycled aluminum beverage can is back on retail shelves within 60 days!"
    },
    "Organic": {
        "icon": "🍎",
        "tagline": "Compostable & Nutrient-Rich Biomass",
        "color": "#2E7D32",
        "badge_bg": "#E8F5E9",
        "bin_name": "Green / Wet / Compost Bin",
        "bin_color": "#2E7D32",
        "steps": [
            "Remove plastic brand stickers, barcode tags, and fruit elastics.",
            "Keep unmixed with plastic wrap, polystyrene, or metallic ties.",
            "Chop thick rinds (e.g., melon, pineapple) to speed decomposition.",
            "Use certified compostable bags or line bins with newspaper."
        ],
        "allowed": [
            "Fruit peels, vegetable peels, scraps, and cores",
            "Coffee grounds, unbleached paper coffee filters, tea leaves",
            "Crushed eggshells and unsalted nutshells",
            "Fallen leaves, garden trimmings, cut flowers, and grass",
            "Leftover cooked plain grains, bread crusts, and rice"
        ],
        "prohibited": [
            "Plastic packaging, cling film, styrofoam food trays",
            "Glossy treated wrappers, plastic stickers, or twist ties",
            "Pet waste or cat litter (contains dangerous pathogens)",
            "Heavy cooking oils, mineral oils, or chemically treated wood"
        ],
        "eco_fact": "When organic waste rots in landfills without oxygen, it creates methane—a greenhouse gas 28x more potent than CO2. Composting turns it into natural fertilizer!"
    },
    "Paper": {
        "icon": "📦",
        "tagline": "Clean Fibers for Renewable Pulp",
        "color": "#0288D1",
        "badge_bg": "#E1F5FE",
        "bin_name": "Blue / Paper Recycling Bin",
        "bin_color": "#0288D1",
        "steps": [
            "Ensure all paper items are clean, dry, and free of food oil.",
            "Flatten and break down cardboard boxes to optimize collection space.",
            "Strip away broad plastic packing tape and internal bubble wrap.",
            "Keep separate from damp or liquid containers to preserve pulp fibers."
        ],
        "allowed": [
            "Corrugated shipping boxes and clean cardboard packaging",
            "Newspapers, magazines, flyers, office printer paper, envelopes",
            "Clean paper grocery bags and cardboard egg cartons",
            "Notebooks (metal spiral removed), paper folders, and cards"
        ],
        "prohibited": [
            "Grease-soaked pizza boxes or oil-stained food wraps",
            "Used paper napkins, tissues, or paper towels (sanitary trash)",
            "Thermal cash register receipts (coated with BPA/BPS)",
            "Wax-coated cups or poly-coated frozen food boxes"
        ],
        "eco_fact": "Recycling one metric ton of paper saves approximately 17 mature trees, 7,000 gallons of fresh water, and 4,000 kWh of electric power!"
    },
    "Plastic": {
        "icon": "🧴",
        "tagline": "Check Resin Code & Divert from Landfill",
        "color": "#7B1FA2",
        "badge_bg": "#F3E5F5",
        "bin_name": "Yellow / Plastic Recyclables Bin",
        "bin_color": "#7B1FA2",
        "steps": [
            "Empty liquids completely and scrape remaining food matter.",
            "Quick rinse with cold water; residue contaminates recycling streams.",
            "Check the triangular resin recycling symbol (#1 PET, #2 HDPE, #5 PP).",
            "Crush plastic bottles and re-attach their caps so they do not get lost."
        ],
        "allowed": [
            "Plastic beverage bottles (PET #1 water, soda, juice bottles)",
            "Detergent jugs, milk containers, shampoo bottles (HDPE #2)",
            "Clean yogurt tubs, margarine containers, plastic food tubs (PP #5)",
            "Rigid, clear plastic containers without food contamination"
        ],
        "prohibited": [
            "Single-use plastic shopping bags, stretch wrap (use store drop-off)",
            "Styrofoam / Expanded Polystyrene food containers (#6 PS)",
            "Multi-layer foil snack bags, candy wrappers, and chip packets",
            "Disposable plastic cutlery, straws, and dirty plastic films"
        ],
        "eco_fact": "Plastics take up to 450+ years to degrade in landfills. Sorting plastics correctly protects marine life, prevents microplastics, and saves petrochemical resources."
    }
}

# -----------------------------------------------------------------------------
# 3. Model Loading & Inference Utilities
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_waste_model():
    """
    Loads the trained Keras model from the project folder.
    Handles Keras 3 format, custom Lambda layers, and safe mode.
    """
    if not os.path.exists(MODEL_FILE):
        return None, f"Model file '{MODEL_FILE}' not found in the project root."

        # First attempt: standard TensorFlow / Keras load
    try:
        import tensorflow as tf

        # The model includes a Lambda layer with 'preprocess_input'
        model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects={"preprocess_input": preprocess_input},
            compile=False,
            safe_mode=False
        )
        return model, None

    except Exception as err1:
        # Second attempt: Keras 3 direct
        try:
            import keras

            model = keras.models.load_model(
                MODEL_FILE,
                custom_objects={"preprocess_input": preprocess_input},
                compile=False,
                safe_mode=False
            )
            return model, None

        except Exception as err2:
            return None, f"Could not load model: {err2}"
       
                
                                

def prepare_image(pil_image: Image.Image) -> np.ndarray:
    """
    Prepares uploaded PIL Image for inference:
    1. Converts image to RGB (stripping transparency/alpha channels).
    2. Resizes image to (224, 224).
    3. Converts to float32 NumPy array with shape (1, 224, 224, 3).
    
    IMPORTANT:
    MobileNetV2 preprocessing is already embedded inside the saved model
    as a Lambda layer. Therefore, we do NOT apply external MobileNetV2
    normalization here; we feed the raw RGB pixel array [0, 255].
    """
    # Convert to RGB mode
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # Resize to exact target input dimensions
    resized = pil_image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    
    # Convert to float32 array
    img_array = np.array(resized, dtype=np.float32)
    
    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


# -----------------------------------------------------------------------------
# 4. User Interface Header & Sidebar
# -----------------------------------------------------------------------------
# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌱 EcoSort AI</div>
    <div class="hero-subtitle">
        Intelligent Deep Learning Waste Classification & Sustainable Disposal Guide.<br>
        Snap or upload a photo of any item to instantly detect its category and get accurate recycling instructions.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🌿 About EcoSort AI")
    st.info(
        "**EcoSort AI** helps individuals, schools, and workplaces separate waste effectively. "
        "By leveraging deep transfer learning with MobileNetV2, it identifies recyclable materials "
        "and guides proper disposal practices to build a zero-waste future."
    )

    st.markdown("---")
    st.markdown("### 🗂️ Target Categories")
    for cat in CLASSES:
        details = WASTE_DETAILS[cat]
        st.markdown(
            f"**{details['icon']} {cat}** &nbsp;•&nbsp; "
            f"<span style='color:{details['color']}; font-weight:600;'>{details['bin_name'].split('/')[0].strip()}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ⚙️ Model Specifications")
    st.markdown("""
    - **Architecture**: MobileNetV2 Transfer Learning
    - **Resolution**: 224 × 224 pixels (RGB)
    - **Internal Preprocessing**: Included in model
    - **Output Classes**: 5 waste types
    """)

    st.markdown("---")
    st.caption("🌍 Every correct sort diverts waste from landfills and saves natural resources.")


# -----------------------------------------------------------------------------
# 5. Main App Navigation Tabs
# -----------------------------------------------------------------------------
tab_classify, tab_directory, tab_tips = st.tabs([
    "📸 Image Classifier",
    "📚 Waste Directory & Disposal Rules",
    "💡 Recycling Best Practices"
])

# -----------------------------------------------------------------------------
# TAB 1: CLASSIFIER
# -----------------------------------------------------------------------------
with tab_classify:
    # Load model
    model, load_error = load_waste_model()

    if load_error:
        st.error(f"⚠️ **Model Initialization Error**: {load_error}")
        st.stop()

    st.markdown("#### 1. Upload or Capture a Waste Item")
    
    # Input selection: File upload or Camera
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload from Device", "📷 Use Web Camera"],
        horizontal=True
    )

    uploaded_file = None
    if input_method == "📁 Upload from Device":
        uploaded_file = st.file_uploader(
            "Select an image file (JPG, JPEG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Take a clear photo centered on the waste object against a neutral background."
        )
    else:
        uploaded_file = st.camera_input("Take a photo of the waste item:")

    # Processing and Display
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Error opening image: {e}")
            st.stop()

        # Display two-column layout: Left = Image, Right = Prediction
        col_img, col_pred = st.columns([1, 1.25], gap="large")

        with col_img:
            st.markdown("##### 🖼️ Uploaded Waste Image")
            st.image(image, use_container_width=True, caption=f"Original Size: {image.size[0]}×{image.size[1]}px")
            
            with st.expander("🔍 Image Preprocessing Details"):
                st.write(f"- **Color Mode**: {image.mode} → RGB")
                st.write(f"- **Target Input Size**: {TARGET_SIZE[0]} × {TARGET_SIZE[1]} px")
                st.write("- **MobileNetV2 Preprocessing**: Performed automatically inside the loaded model.")

        with col_pred:
            st.markdown("##### 🎯 Classification Results")

            # Run inference
            with st.spinner("Analyzing waste material with AI model..."):
                input_tensor = prepare_image(image)
                raw_predictions = model.predict(input_tensor, verbose=0)[0]
                
                # Softmax probabilities if not already normalized
                if np.sum(raw_predictions) > 0:
                    probabilities = raw_predictions / np.sum(raw_predictions)
                else:
                    probabilities = raw_predictions

                # Find top prediction
                top_index = int(np.argmax(probabilities))
                predicted_category = CLASSES[top_index]
                confidence_pct = float(probabilities[top_index] * 100)

            # Details of top prediction
            cat_data = WASTE_DETAILS[predicted_category]

            # Top result banner
            st.markdown(f"""
            <div class="result-card" style="border-color: {cat_data['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span class="badge-category" style="background-color: {cat_data['badge_bg']}; color: {cat_data['color']};">
                        {cat_data['icon']} {predicted_category}
                    </span>
                    <span class="badge-confidence">
                        ⚡ {confidence_pct:.1f}% Confidence
                    </span>
                </div>
                <div style="color: #455a64; font-size: 0.95rem; margin-bottom: 0.8rem;">
                    <strong>Designated Bin:</strong> 
                    <span style="color: {cat_data['bin_color']}; font-weight: 700;">{cat_data['bin_name']}</span>
                </div>
                <div style="font-size: 0.88rem; color: #2e7d32; font-weight: 600;">
                    ✓ {cat_data['tagline']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📊 Class Probabilities Breakdown")

            # Horizontal probability bars
            for idx, cat_name in enumerate(CLASSES):
                prob = float(probabilities[idx] * 100)
                item_color = WASTE_DETAILS[cat_name]["color"]
                is_winner = (idx == top_index)

                st.markdown(f"""
                <div class="prob-bar-container">
                    <div class="prob-bar-label">
                        <span style="font-weight: {'700' if is_winner else '500'}; color: {item_color if is_winner else '#37474f'};">
                            {WASTE_DETAILS[cat_name]['icon']} {cat_name} {'⭐' if is_winner else ''}
                        </span>
                        <span style="font-weight: 700; color: {item_color if is_winner else '#546e7a'};">
                            {prob:.2f}%
                        </span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {max(prob, 1.0)}%; background-color: {item_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # Full Waste Disposal Guidance for Predicted Category
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown(f"### 📋 Tailored Disposal Guidance for {cat_data['icon']} **{predicted_category}**")

        col_g1, col_g2 = st.columns([1, 1], gap="large")

        with col_g1:
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: #1b5e20; margin-top: 0;">🛠️ Preparation Checklist</h4>
                <p style="color: #546e7a; font-size: 0.9rem;">Follow these steps before placing in the bin:</p>
            """, unsafe_allow_html=True)

            for step in cat_data["steps"]:
                st.markdown(f"- ✅ **{step}**")

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="eco-tip-box">
                <strong>🌍 Environmental Impact:</strong><br>
                {cat_data['eco_fact']}
            </div>
            """, unsafe_allow_html=True)

        with col_g2:
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: #2e7d32; margin-top: 0;">✅ Accepted Items</h4>
            """, unsafe_allow_html=True)
            for item in cat_data["allowed"]:
                st.markdown(f"""<div class="do-box">✓ {item}</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: #c62828; margin-top: 0;">❌ Items to Keep Out</h4>
            """, unsafe_allow_html=True)
            for item in cat_data["prohibited"]:
                st.markdown(f"""<div class="dont-box">✕ {item}</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Placeholder banner when no image is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; border: 2px dashed #a5d6a7; border-radius: 16px; background-color: #f9fbf9; margin-top: 1rem;">
            <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">📸</div>
            <h3 style="color: #2e7d32; margin: 0.2rem 0;">No Waste Image Uploaded Yet</h3>
            <p style="color: #689f38; max-width: 500px; margin: 0.5rem auto 1.5rem auto; font-size: 0.95rem;">
                Upload a photo or capture an image of any waste item above to identify its category and receive step-by-step eco-friendly disposal guidelines.
            </p>
            <div style="display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap;">
                <span style="background: #e8f5e9; padding: 0.35rem 0.8rem; border-radius: 20px; font-size: 0.85rem; color: #1b5e20;">🍾 Glass</span>
                <span style="background: #fff3e0; padding: 0.35rem 0.8rem; border-radius: 20px; font-size: 0.85rem; color: #e65100;">🥫 Metal</span>
                <span style="background: #e8f5e9; padding: 0.35rem 0.8rem; border-radius: 20px; font-size: 0.85rem; color: #2e7d32;">🍎 Organic</span>
                <span style="background: #e1f5fe; padding: 0.35rem 0.8rem; border-radius: 20px; font-size: 0.85rem; color: #0288d1;">📦 Paper</span>
                <span style="background: #f3e5f5; padding: 0.35rem 0.8rem; border-radius: 20px; font-size: 0.85rem; color: #7b1fa2;">🧴 Plastic</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: DIRECTORY OF ALL CATEGORIES
# -----------------------------------------------------------------------------
with tab_directory:
    st.markdown("### 📚 Waste Sorting Knowledge Base")
    st.write("Browse disposal guidelines, bin requirements, and eco tips for each of the five waste categories.")

    selected_category = st.selectbox(
        "Select category to inspect:",
        CLASSES,
        format_func=lambda x: f"{WASTE_DETAILS[x]['icon']} {x} — {WASTE_DETAILS[x]['tagline']}"
    )

    spec = WASTE_DETAILS[selected_category]

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown(f"""
        <div class="custom-card">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 2.2rem;">{spec['icon']}</span>
                <div>
                    <h3 style="margin: 0; color: {spec['color']};">{selected_category} Waste</h3>
                    <div style="color: #546e7a; font-size: 0.9rem;">{spec['tagline']}</div>
                </div>
            </div>
            <hr style="margin: 1rem 0; border: none; border-top: 1px solid #edf2ee;">
            <p><strong>Recommended Bin:</strong> <span style="color: {spec['bin_color']}; font-weight: 700;">{spec['bin_name']}</span></p>
            <h5 style="color: #1b5e20;">Preparation Steps:</h5>
        """, unsafe_allow_html=True)

        for s in spec["steps"]:
            st.markdown(f"- {s}")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="eco-tip-box">
            <strong>🌱 Ecological Fact:</strong><br>
            {spec['eco_fact']}
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="custom-card">
            <h4 style="color: #2e7d32; margin-top: 0;">✅ Accepted Items</h4>
        """, unsafe_allow_html=True)
        for itm in spec["allowed"]:
            st.markdown(f"""<div class="do-box">✓ {itm}</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="custom-card">
            <h4 style="color: #c62828; margin-top: 0;">❌ Keep Out / Contaminants</h4>
        """, unsafe_allow_html=True)
        for itm in spec["prohibited"]:
            st.markdown(f"""<div class="dont-box">✕ {itm}</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 3: RECYCLING TIPS & CIRCULAR ECONOMY
# -----------------------------------------------------------------------------
with tab_tips:
    st.markdown("### 💡 The Golden Rules of Waste Management")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.markdown("""
        <div class="custom-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚿</div>
            <h4 style="color: #1b5e20; margin: 0 0 0.5rem 0;">1. Clean & Dry</h4>
            <p style="color: #455a64; font-size: 0.9rem; line-height: 1.5;">
                Even a small amount of leftover food or grease can spoil an entire batch of recyclable paper or plastic. Give containers a quick rinse and let them dry before tossing.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        st.markdown("""
        <div class="custom-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📦</div>
            <h4 style="color: #1b5e20; margin: 0 0 0.5rem 0;">2. Flatten & Compact</h4>
            <p style="color: #455a64; font-size: 0.9rem; line-height: 1.5;">
                Break down corrugated cardboard boxes and gently crush aluminum beverage cans. Compacting materials reduces collection volume and lowers transport fuel consumption.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r3:
        st.markdown("""
        <div class="custom-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚫</div>
            <h4 style="color: #1b5e20; margin: 0 0 0.5rem 0;">3. Avoid "Wish-Cycling"</h4>
            <p style="color: #455a64; font-size: 0.9rem; line-height: 1.5;">
                Wish-cycling is tossing non-recyclable items into the bin hoping they will be recycled. Contaminants jam sorting machinery and send entire truckloads to landfills.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-card" style="margin-top: 1rem; border-left: 5px solid #2e7d32;">
        <h4 style="color: #1b5e20; margin-top: 0;">♻️ The Waste Hierarchy: Reduce &gt; Reuse &gt; Recycle</h4>
        <p style="color: #37474f; font-size: 0.95rem; line-height: 1.6;">
            While recycling is vital, the most sustainable waste is the waste that is never created in the first place:
        </p>
        <ul style="color: #455a64; line-height: 1.8;">
            <li><strong>Reduce:</strong> Choose minimal packaging, avoid single-use plastics, and opt for durable goods.</li>
            <li><strong>Reuse:</strong> Repurpose glass jars for food storage, carry reusable tote bags, and donate old items.</li>
            <li><strong>Recycle:</strong> Use EcoSort AI to ensure every item reaches the proper recycling stream!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("<br><hr style='border: none; border-top: 1px solid #e0e7e1;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #689f38; font-size: 0.88rem; padding: 0.5rem 0 1.5rem 0;">
    🌱 <strong>EcoSort AI</strong> • Powered by TensorFlow & Streamlit • Dedicated to a Cleaner, Greener Planet.
</div>
""", unsafe_allow_html=True)
