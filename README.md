# 🧠 GPU Memory Calculator for LLMs

A Streamlit web application that helps you estimate GPU memory requirements for Hugging Face models. Original gist by [@philschmid](https://gist.github.com/philschmid).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/gabzofar/llm-gpu-memory-calculator)

## Features

- Search Hugging Face models by name
- Quick search buttons for popular model families
- Support for different data types (float16, bfloat16, float32)
- Real-time memory requirement calculations
- User-friendly interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gabzofar/LLM-GPU-Memory-Calculator.git
cd LLM-GPU-Memory-Calculator
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser. You can then:
1. Search for models using the search bar
2. Use quick search buttons for popular model families
3. Select a specific model from the search results
4. Choose your desired data type
5. Click "Calculate Memory Requirements" to see the estimated GPU memory needed

## Note

- Memory estimates include an 18% overhead for CUDA kernels and runtime requirements
- Actual memory usage may vary depending on your specific setup
- Memory calculations use binary prefix (1024³ bytes per GiB)

## Credits

- Original concept: [@philschmid](https://gist.github.com/philschmid)
- Built with [Streamlit](https://streamlit.io)
- Model data from [🤗 Hugging Face](https://huggingface.co) 