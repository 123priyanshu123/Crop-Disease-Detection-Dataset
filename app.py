import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern agro-tech UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1e4620;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b6b4e;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background-color: #f4fbf5;
        border: 1px solid #c2e5c8;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .healthy-badge {
        display: inline-block;
        background-color: #2e7d32;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .disease-badge {
        display: inline-block;
        background-color: #c62828;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .remedy-box {
        background-color: #ffffff;
        border-left: 5px solid #2e7d32;
        padding: 15px 20px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & CLASS NAMES (29 Classes)
# ==========================================
CLASS_NAMES = [
    'Apple - Apple Scab',
    'Apple - Black Rot',
    'Apple - Cedar Apple Rust',
    'Apple - Healthy',
    'Bell Pepper - Bacterial Spot',
    'Bell Pepper - Healthy',
    'Cherry - Healthy',
    'Cherry - Powdery Mildew',
    'Corn (Maize) - Cercospora Leaf Spot',
    'Corn (Maize) - Common Rust',
    'Corn (Maize) - Healthy',
    'Corn (Maize) - Northern Leaf Blight',
    'Grape - Black Rot',
    'Grape - Esca (Black Measles)',
    'Grape - Healthy',
    'Grape - Leaf Blight',
    'Peach - Bacterial Spot',
    'Peach - Healthy',
    'Potato - Early Blight',
    'Potato - Healthy',
    'Potato - Late Blight',
    'Strawberry - Healthy',
    'Strawberry - Leaf Scorch',
    'Tomato - Bacterial Spot',
    'Tomato - Early Blight',
    'Tomato - Healthy',
    'Tomato - Late Blight',
    'Tomato - Septoria Leaf Spot',
    'Tomato - Yellow Leaf Curl Virus'
]

# Knowledge base for disease remedies & care
DISEASE_DETAILS = {
    'Healthy': {
        'symptoms': 'The foliage exhibits uniform vibrant color, intact cellular structure, and no visible lesions, chlorosis, or necrotic tissue.',
        'organic': 'Maintain regular irrigation, ensure optimal soil aeration, and feed with balanced compost or organic nitrogen-phosphorus-potassium (NPK).',
        'chemical': 'No chemical fungicide or bactericide intervention required. Continue preventive pest scouting.',
        'prevention': 'Ensure proper plant spacing for airflow, practice crop rotation, and inspect undersides of leaves weekly.'
    },
    'Apple Scab': {
        'symptoms': 'Olive-green to velvety dark brown spots on leaves, becoming puckered and distorted; velvety lesions on fruit.',
        'organic': 'Apply sulfur sprays or neem oil early in the spring. Remove and compost or burn fallen leaves during autumn to break pathogen cycle.',
        'chemical': 'Fungicides containing Captan, Mancozeb, or Myclobutanil applied from bud break through petal fall.',
        'prevention': 'Prune trees to open canopy for sun and air penetration. Choose scab-resistant apple cultivars.'
    },
    'Black Rot': {
        'symptoms': 'Brown circular leaf lesions expanding with tiny black pycnidia; mummified black shriveled fruit on tree or vine.',
        'organic': 'Prune out dead wood, mummified fruits, and cankers during dormant season. Disinfect pruning shears between cuts.',
        'chemical': 'Copper-based fungicides, Captan, or Thiophanate-methyl applications before infection periods.',
        'prevention': 'Avoid overhead watering; remove infected plant debris from the orchard floor.'
    },
    'Cedar Apple Rust': {
        'symptoms': 'Bright yellow-orange spots on upper leaf surfaces; tube-like aecia structures forming on leaf undersides in mid-summer.',
        'organic': 'Apply preventive copper or sulfur spray early in spring. Remove nearby Eastern red cedar galls within a 1-mile radius if possible.',
        'chemical': 'Myclobutanil or Propiconazole applied when cedar galls are actively producing gelatinous horns.',
        'prevention': 'Plant rust-resistant cultivars (e.g., Enterprise, Freedom, Liberty).'
    },
    'Bacterial Spot': {
        'symptoms': 'Small, angular, water-soaked dark spots with yellow halos on foliage; spots may tear or drop out creating a "shot-hole" look.',
        'organic': 'Apply copper octanoate (copper soap) spray at the earliest signs. Avoid working in fields when foliage is wet.',
        'chemical': 'Copper hydroxide bactericides combined with Mancozeb for improved bacterial control.',
        'prevention': 'Use certified pathogen-free seeds and disease-resistant transplants. Rotate with non-solanaceous crops for 2-3 years.'
    },
    'Powdery Mildew': {
        'symptoms': 'White powdery fungal coating on leaf surfaces, young shoots, and flower buds; leaves may curl upward and drop prematurely.',
        'organic': 'Spray with potassium bicarbonate, dilute milk spray (40% milk, 60% water), or horticultural oils.',
        'chemical': 'Sulfur-based fungicides, Trifloxystrobin, or Myclobutanil.',
        'prevention': 'Plant in full sun, avoid high nitrogen fertilizer that promotes tender susceptible growth, and space plants generously.'
    },
    'Cercospora Leaf Spot': {
        'symptoms': 'Small grayish-tan circular or oval lesions bounded by dark purplish-brown borders on leaves.',
        'organic': 'Bio-fungicide sprays containing Bacillus subtilis. Incorporate or shred crop residue after harvest.',
        'chemical': 'Azoxystrobin, Pyraclostrobin, or Chlorothalonil applied when conditions favor disease development.',
        'prevention': 'Practice crop rotation and avoid planting continuous corn. Plant resistant hybrids.'
    },
    'Common Rust': {
        'symptoms': 'Golden-brown to cinnamon-brown powdery pustules (uredinia) scattered on both upper and lower leaf surfaces.',
        'organic': 'Apply sulfur dust or neem oil extract on initial identification. Early harvest can limit silage loss.',
        'chemical': 'Foliar triazole or strobilurin fungicides (e.g., Headline, Quadris) if rust spreads prior to tassel stage.',
        'prevention': 'Select corn hybrids with genetic resistance (Rp genes).'
    },
    'Northern Leaf Blight': {
        'symptoms': 'Long, elliptical, grayish-green or cigar-shaped lesions (1 to 6 inches long) that turn tan with fungal sporulation.',
        'organic': 'Deep tillage of infected debris into soil, and application of bio-rational fungicides.',
        'chemical': 'Apply Strobilurin + Triazole premix fungicides when lesions reach ear leaf before pollination.',
        'prevention': 'Utilize 2-year non-host crop rotation and select tolerant corn hybrids.'
    },
    'Esca (Black Measles)': {
        'symptoms': 'Interveinal "tiger-stripe" chlorosis and necrosis on grape leaves; dark spots or measles on berries.',
        'organic': 'Paint pruning wounds with wound protectants (paste with Trichoderma spp.). Prune in late winter dry periods.',
        'chemical': 'No effective systemic chemical curative; prevent wound infection with registered fungicidal sealants.',
        'prevention': 'Avoid creating large pruning wounds in damp weather; replace severely affected vines.'
    },
    'Leaf Blight': {
        'symptoms': 'Irregular brown necrotic patches on leaves, often starting at leaf margins and spreading inward.',
        'organic': 'Apply copper fungicides or bio-fungicides; clean out under-vine canopy debris.',
        'chemical': 'Captan, Mancozeb, or Boscalid formulations applied during early shoot growth.',
        'prevention': 'Canopy management (leaf pulling) to ensure maximum airflow and sunlight exposure.'
    },
    'Early Blight': {
        'symptoms': 'Dark brown to black lesions featuring characteristic concentric rings resembling a "target-board", surrounded by yellow halo.',
        'organic': 'Copper fungicide or Bacillus amyloliquefaciens spray every 7-10 days. Prune lower leaves touching soil.',
        'chemical': 'Chlorothalonil, Mancozeb, Difenoconazole, or Azoxystrobin applications.',
        'prevention': 'Mulch around base to prevent soil splash; practice drip irrigation; rotate crops every 3 seasons.'
    },
    'Late Blight': {
        'symptoms': 'Water-soaked irregular pale-green to dark brown patches rapidly enlarging; white cottony fungal growth on leaf undersides in humid conditions.',
        'organic': 'Immediate preventative copper hydroxide sprays; eradicate and safely bag infected plants to prevent airborne spread.',
        'chemical': 'Metalaxyl, Mefenoxam, Cyazofamid, or Mandipropamid as preventative and curative regimens.',
        'prevention': 'Plant certified blight-free seed potatoes and disease-resistant tomato varieties. Destroy volunteer potato plants.'
    },
    'Leaf Scorch': {
        'symptoms': 'Purplish-brown circular blotches on leaves that enlarge and turn dark brown with dry, scorched leaf edges.',
        'organic': 'Remove infected foliage after fruiting; spray bio-fungicides or neem oil solution.',
        'chemical': 'Captan or Myclobutanil applied during renewal or when new leaf flushes emerge.',
        'prevention': 'Renovate strawberry beds after harvest; avoid overhead sprinkler irrigation in late afternoon.'
    },
    'Septoria Leaf Spot': {
        'symptoms': 'Numerous small circular spots with dark brown margins and light gray centers, studded with tiny black fruiting specks.',
        'organic': 'Strip bottom 12 inches of foliage as plants mature; spray copper octanoate or Serenade bio-fungicide.',
        'chemical': 'Chlorothalonil or copper-based fungicides applied as soon as bottom spots appear.',
        'prevention': 'Stake or cage tomatoes; apply organic mulch (straw/leaves); sanitize garden stakes each season.'
    },
    'Yellow Leaf Curl Virus': {
        'symptoms': 'Severe upward cupping and crumpling of leaves; intense yellowing (chlorosis) of leaf margins; stunted bushy plant growth.',
        'organic': 'Control silverleaf whitefly vectors using yellow sticky traps, insecticidal soap, and reflective silver mulches.',
        'chemical': 'Imidacloprid or Pyrethroid insecticide treatments targeting whitefly nymphs and adults.',
        'prevention': 'Use fine insect netting (50 mesh); destroy infected plants immediately; plant TYLCV-resistant varieties.'
    }
}

# ==========================================
# MODEL SETUP & LOADING
# ==========================================
@st.cache_resource
def load_pytorch_model():
    """
    Initializes ResNet-18 model architecture and loads weights if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)
    
    # Initialize ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    # Match the Sequential(Dropout, Linear) architecture from training
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    weights_path = "plant_disease_model.pth"
    is_trained = False
    
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            # Support both full state_dict and nested checkpoint formats
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            is_trained = True
        except Exception:
            # Fallback if checkpoint was saved with single Linear layer
            try:
                model.fc = nn.Linear(in_features, num_classes)
                model.load_state_dict(state_dict)
                is_trained = True
            except Exception as e:
                st.sidebar.warning(f"Note loading custom checkpoint: {e}. Using initialized model.")
                is_trained = False
            
    model = model.to(device)
    model.eval()
    return model, device, is_trained

# ==========================================
# INFERENCE PIPELINE
# ==========================================
def predict_leaf(image: Image.Image, model, device):
    """
    Runs image preprocessing and model forward pass to output predictions.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
    top_probs, top_indices = torch.topk(probabilities, k=min(5, len(CLASS_NAMES)))
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class_name": CLASS_NAMES[idx.item()],
            "confidence": prob.item() * 100.0,
            "index": idx.item()
        })
        
    return results

def get_remedy_info(class_name: str):
    """
    Extracts relevant remedy advice based on class name.
    """
    if "Healthy" in class_name:
        return DISEASE_DETAILS['Healthy'], True
    
    for key in DISEASE_DETAILS:
        if key in class_name:
            return DISEASE_DETAILS[key], False
            
    # Generic fallback
    return {
        'symptoms': 'Visible leaf discoloration or pathogen spotting detected.',
        'organic': 'Isolate affected plant, prune symptomatic leaves, apply neem oil or copper spray.',
        'chemical': 'Consult local agricultural extension for registered broad-spectrum fungicides.',
        'prevention': 'Maintain sanitation, avoid overhead watering, ensure generous plant spacing.'
    }, False

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.title("🌱 PlantVillage AI")
    st.markdown("""
    **Crop Health & Pathology Diagnosis**  
    Deep Learning system powered by **PyTorch ResNet-18** to automatically detect and classify crop diseases across 29 categories.
    """)
    
    st.divider()
    
    model, device, is_trained = load_pytorch_model()
    
    st.subheader("⚙️ System Status")
    st.write(f"**Execution Device:** `{device.type.upper()}`")
    if is_trained:
        st.success("✅ Trained Model Weights Loaded (`plant_disease_model.pth`)")
    else:
        st.info("ℹ️ Running PyTorch Transfer Learning Architecture (`ResNet-18`). Train via `Crop_Disease_Detection.ipynb` to save fine-tuned weights.")
        
    st.divider()
    
    with st.expander("🌿 Supported Crops & Classes (29)"):
        crops = {}
        for c in CLASS_NAMES:
            crop_name = c.split(" - ")[0]
            condition = c.split(" - ")[1]
            crops.setdefault(crop_name, []).append(condition)
        
        for crop, conds in sorted(crops.items()):
            st.markdown(f"**{crop}** ({len(conds)}):")
            for cond in conds:
                icon = "🟢" if "Healthy" in cond else "🔴"
                st.markdown(f"- {icon} {cond}")

# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-title">🌿 Crop Disease Diagnosis & Management</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a leaf image or select from the benchmark test dataset to inspect plant health and receive instant organic and chemical treatment recommendations.</div>', unsafe_allow_html=True)

# Layout: 2 Columns for Input & Results
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("1. Select Leaf Image")
    
    input_source = st.radio(
        "Choose Input Method:",
        ["📤 Upload Leaf Image", "📁 Select from Test Dataset"],
        horizontal=True
    )
    
    selected_image = None
    image_title = ""
    
    if input_source == "📤 Upload Leaf Image":
        uploaded_file = st.file_uploader(
            "Upload leaf photograph (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file)
                image_title = uploaded_file.name
            except Exception as e:
                st.error(f"Error opening image: {e}")
                
    else:
        # Load sample from Test directory
        test_dir = "Test"
        if os.path.exists(test_dir):
            categories = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
            selected_category = st.selectbox("Select Disease Category:", categories)
            
            cat_path = os.path.join(test_dir, selected_category)
            available_files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if available_files:
                selected_file = st.selectbox("Select Sample Image:", available_files)
                img_path = os.path.join(cat_path, selected_file)
                selected_image = Image.open(img_path)
                image_title = f"{selected_category} ({selected_file})"
            else:
                st.warning("No image files found in selected category directory.")
        else:
            st.warning("`Test` directory not found. Please upload an image directly.")
            
    if selected_image is not None:
        st.image(selected_image, caption=f"Selected: {image_title}", use_container_width=True)
        st.caption(f"Image Resolution: {selected_image.width} × {selected_image.height} px | Format: {selected_image.format or 'RGB'}")

with col_result:
    st.subheader("2. Diagnostic Results & Insights")
    
    if selected_image is not None:
        with st.spinner("Analyzing leaf pathology with PyTorch deep neural network..."):
            predictions = predict_leaf(selected_image, model, device)
            
        top_pred = predictions[0]
        remedy_data, is_healthy = get_remedy_info(top_pred["class_name"])
        
        # Split crop and disease name
        parts = top_pred["class_name"].split(" - ")
        crop_name = parts[0]
        disease_name = parts[1] if len(parts) > 1 else top_pred["class_name"]
        
        # Display top diagnosis card
        badge_html = (
            f'<span class="healthy-badge">🟢 HEALTHY CROP</span>'
            if is_healthy else
            f'<span class="disease-badge">🔴 PATHOLOGY DETECTED</span>'
        )
        
        st.markdown(f"""
        <div class="prediction-card">
            {badge_html}
            <h2 style="color: #1e4620; margin-top: 10px; margin-bottom: 5px;">{top_pred['class_name']}</h2>
            <p style="font-size: 1.15rem; color: #333; margin-bottom: 0;">
                Confidence: <strong>{top_pred['confidence']:.2f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability distribution for Top 3
        st.markdown("#### Top Prediction Probabilities")
        for pred in predictions[:3]:
            col_bar_name, col_bar_val = st.columns([3, 1])
            with col_bar_name:
                st.write(f"**{pred['class_name']}**")
            with col_bar_val:
                st.write(f"{pred['confidence']:.2f}%")
            st.progress(min(1.0, max(0.0, pred['confidence'] / 100.0)))
            
        st.divider()
        
        # Actionable Remedies & Care Guide
        st.markdown("#### 📋 Recommended Treatment & Agronomic Actions")
        
        st.markdown(f"**🔍 Pathology Symptoms:**")
        st.info(remedy_data['symptoms'])
        
        tab_org, tab_chem, tab_prev = st.tabs(["🌱 Organic Management", "🧪 Chemical Treatment", "🛡️ Cultural Prevention"])
        
        with tab_org:
            st.markdown(remedy_data['organic'])
        with tab_chem:
            st.markdown(remedy_data['chemical'])
        with tab_prev:
            st.markdown(remedy_data['prevention'])
            
    else:
        st.info("👈 Please upload an image or choose a sample from the test dataset to run diagnosis.")
        
        # Quick overview cards
        st.markdown("### 🌾 About the Detection Pipeline")
        st.markdown("""
        - **PyTorch Backbone:** Fine-tuned Deep Convolutional Neural Network (`ResNet-18`).
        - **Dataset:** Over 53,000 training images across 29 classes from the PlantVillage dataset.
        - **Preprocessing:** Standard ImageNet channel normalization, automated aspect-ratio resizing to $224 \\times 224$.
        - **Early Diagnostics:** Helps prevent severe crop loss through timely disease classification and targeted treatment plans.
        """)

st.divider()
st.caption("Plant Village Crop Disease Detection | PyTorch & Streamlit | Precision Agriculture Initiative")
