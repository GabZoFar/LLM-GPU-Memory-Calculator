import streamlit as st
from huggingface_hub import HfApi
from get_memory_size import get_model_size, bytes_per_dtype

st.set_page_config(
    page_title="GPU Memory Calculator",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 GPU Memory Calculator for LLMs")
st.write("Estimate GPU memory requirements for Hugging Face models")

# Initialize Hugging Face API
hf_api = HfApi()

# Create placeholder for status messages
status_placeholder = st.empty()

# Search functionality
search_query = st.text_input("Search for models", placeholder="Enter model name (e.g., llama, gpt)")

# Add quick search buttons
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    if st.button("Nvidia"):
        search_query = "Nvidia"
with col2:
    if st.button("nemotron"):
        search_query = "nemotron"
with col3:
    if st.button("DeepSeek"):
        search_query = "DeepSeek"
with col4:
    if st.button("mistral"):
        search_query = "mistral"
with col5:
    if st.button("bartowski"):
        search_query = "bartowski"
with col6:
    if st.button("llama"):
        search_query = "llama"

if search_query:
    # Cache the search results to avoid rate limiting
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def search_models(query):
        models = hf_api.list_models(
            search=query,
            limit=50,
            sort="downloads",
            direction=-1
        )
        # Convert generator to list of dictionaries with only the data we need
        return [{"id": model.id, "downloads": model.downloads} for model in models]
    
    models = search_models(search_query)
    
    if models:
        # Sort models by downloads (most popular first)
        models.sort(key=lambda x: x["downloads"] if x["downloads"] is not None else 0, reverse=True)
        model_ids = [model["id"] for model in models]
        
        # Initialize session state for selected values if they don't exist
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_ids[0]
        if 'selected_dtype' not in st.session_state:
            st.session_state.selected_dtype = "float16"
        
        # Create a form to prevent automatic reruns
        with st.form("model_form"):
            # Create two columns for model and dtype selection
            col1, col2 = st.columns(2)
            
            with col1:
                selected_model = st.selectbox(
                    "Select a model",
                    model_ids,
                    index=model_ids.index(st.session_state.selected_model)
                )
            
            with col2:
                dtype = st.selectbox(
                    "Select data type",
                    list(bytes_per_dtype.keys()),
                    index=list(bytes_per_dtype.keys()).index(st.session_state.selected_dtype)
                )

            # Add dtype explanations
            st.markdown("""
            **Data Type Information:**
            - **float16**: Often used in GPU training to save memory while maintaining reasonable precision
            - **bfloat16**: Popular in neural networks as it has a better dynamic range
            - **float32**: Standard choice when memory isn't a constraint and you need good precision
            """)
            
            # Replace button with form submit button
            submitted = st.form_submit_button("Calculate Memory Requirements")
            
            if submitted:
                # Update session state
                st.session_state.selected_model = selected_model
                st.session_state.selected_dtype = dtype
                
                # Show progress through multiple status updates
                status_placeholder.info("Starting calculation...")
                status_placeholder.code("model.safetensors.index.json: Downloading metadata...")
                
                memory_size = get_model_size(selected_model, dtype)
                
                if memory_size is not None:
                    status_placeholder.success("✅ Calculation complete!")
                    st.success(f"Estimated GPU memory requirement: **{memory_size:.2f} GiB** ({dtype})")
                    
                    st.info("""
                    📝 **Note:**
                    - This is an estimate that includes an 18% overhead for CUDA kernels and runtime requirements
                    - Actual memory usage may vary depending on your specific setup and usage
                    - Memory calculation uses binary prefix (1024³ bytes per GiB)
                    """)
                else:
                    status_placeholder.error("❌ Could not calculate memory requirements. The model might not have proper metadata.")
    else:
        st.warning("No models found matching your search query.")

# Footer
st.markdown("---")
st.markdown(
    "Orignal Gist from [philschmid](https://gist.github.com/philschmid). Made with ❤️ by [Gabrielle](https://github.com/gabzofar) using [Streamlit](https://streamlit.io) | "
    "Data from [🤗 Hugging Face](https://huggingface.co)"
)