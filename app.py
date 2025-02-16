import sys
sys.modules['__main__'].__file__ = 'app.py'  # Fix Streamlit path detection

import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import time

# ======================
# Free Model Setup
# ======================

@st.cache_resource
def load_story_model():
    """Load free text generation model"""
    try:
        return pipeline("text-generation", model="gpt2-medium")
    except Exception as e:
        st.error(f"Error loading story model: {str(e)}")
        return None

@st.cache_resource 
def load_image_model():
    """Load free Stable Diffusion model"""
    try:
        model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_auth_token=False  # No authentication needed
        )
        return model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        st.error(f"Error loading image model: {str(e)}")
        return None

# ======================
# Core Functions
# ======================

def generate_story(prompt, story_gen, max_length=200):
    try:
        result = story_gen(
            prompt,
            max_length=max_length,
            truncation=True,  # Explicitly enable truncation
            pad_token_id=story_gen.tokenizer.eos_token_id,  # Set pad token
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
        return result[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

def generate_image(prompt, image_pipe):
    try:
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            # Add progress callback
            def callback(step, total_steps, latents):
                progress = step / total_steps
                st.write(f"Generating image: {progress:.0%}")

            image = image_pipe(
                prompt,
                callback=callback,
                callback_steps=5  # Update every 5 steps
            ).images[0]
        return image
    except Exception as e:
        return None

# ======================
# Streamlit UI
# ======================

st.title("🧠 Free NeuroStoryteller")
st.markdown("### (No API Keys Required)")

# Load models
story_gen = load_story_model()
image_pipe = load_image_model()

# User Input
prompt = st.text_input("Enter your story prompt:")
max_length = st.slider("Max length", 100, 500, 200)

if prompt and story_gen and image_pipe:
    with st.spinner("Generating your free story..."):
        start_time = time.time()
        
        # Generate story
        story = generate_story(prompt, story_gen, max_length)
        
        # Generate image
        image = generate_image(prompt, image_pipe)
        
        st.subheader("Generated Story")
        st.write(story)
        
        if image:
            st.subheader("Generated Image")
            st.image(image, caption=prompt)
            
        st.info(f"Total generation time: {time.time()-start_time:.1f}s")
        
elif not story_gen or not image_pipe:
    st.error("Failed to load required models. Check your internet connection!")