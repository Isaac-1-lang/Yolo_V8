# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
from category_mapper import WasteCategoryMapper

# Setting page layout
st.set_page_config(
    page_title="Waste Classification using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("🗑️ Waste Classification using YOLOv8")

# Initialize category mapper
mapper = WasteCategoryMapper()
desired_categories = mapper.get_desired_categories()

st.markdown(f"**Classify waste into: {', '.join(desired_categories)}**")
st.markdown("*Note: Model categories are mapped to your desired waste categories*")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("🔍 Detection Results"):
                        if len(boxes) > 0:
                            # Map predictions to desired categories
                            mapped_results = mapper.map_prediction(model, res[0])
                            
                            st.write(f"**Detected {len(mapped_results)} waste item(s):**")
                            
                            # Create a summary of detected categories (mapped)
                            category_counts = {}
                            for item in mapped_results:
                                mapped_class = item['mapped_class']
                                if mapped_class not in category_counts:
                                    category_counts[mapped_class] = []
                                category_counts[mapped_class].append(item['confidence'])
                            
                            # Display category summary
                            st.write("**Waste Categories (Mapped):**")
                            for category, confidences in category_counts.items():
                                avg_conf = sum(confidences) / len(confidences)
                                st.write(f"• **{category}**: {len(confidences)} item(s) (avg confidence: {avg_conf:.2f})")
                            
                            # Display detailed results
                            st.write("\n**Detailed Results:**")
                            for i, item in enumerate(mapped_results):
                                st.write(f"{i+1}. **{item['mapped_class']}** (was {item['original_class']}): {item['confidence']:.3f}")
                            
                            # Show mapping info
                            with st.expander("ℹ️ Category Mapping Info"):
                                mapping_info = mapper.get_mapping_info()
                                st.write("**How model categories are mapped to your desired categories:**")
                                for original, desired in mapping_info.items():
                                    st.write(f"• {original} → {desired}")
                        else:
                            st.write("No waste items detected in the image.")
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")


