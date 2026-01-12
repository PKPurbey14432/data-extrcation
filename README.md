# ML-Based Product Automation System

A comprehensive ML automation system for product processing that includes:
- **OCR Text Extraction** (vertical, horizontal, and embossed text)
- **Background Removal** from product images
- **Attribute Extraction** using LLM from product title, description, and images

## Features

### 1. OCR Text Extraction
- Extracts text from images in multiple orientations (horizontal and vertical)
- Handles embossed/raised text with specialized preprocessing
- Uses multiple OCR engines (EasyOCR and PaddleOCR) for better accuracy
- Returns confidence scores and bounding box information

### 2. Background Removal
- Removes backgrounds from product images using deep learning
- Supports transparency (PNG output)
- Uses state-of-the-art U2Net models

### 3. Attribute Extraction
- Extracts comprehensive product attributes using LLM (OpenAI GPT-4)
- Analyzes product title, description, and image text
- Supports vision models for direct image analysis
- Returns structured JSON with categorized attributes

## Installation

### Prerequisites
- Python 3.10 or higher
- pip3

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd /home/lnv221/mine/lumina
```
2. **Create VirtualENV:**
```bash
python3.10 -m venv venv
```

3. **activate venv:**
```bash
venv/bin/activate
```

4. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

5. **Set up API keys (for attribute extraction):**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

1. OCR Text Extraction Only
2. Background Removal Only
3. Attribute Extraction Only


## for Test
update the image path in each file
```bash
python attribute_extractor.py
python background_remover.py
python ocr_extractor.py

```
