# NewsBot 2.0 — Deployment Guide

## Local Development Setup

### Prerequisites
- Python 3.10+
- Git
- [ollama](https://ollama.ai) (for LLM features)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Student-Portfolio-Repository.git
cd Student-Portfolio-Repository/ITAI2373-NewsBot-Final

# 2. Create virtual environment (recommended)
python -m venv newsbot_env
source newsbot_env/bin/activate       # Mac/Linux
newsbot_env\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Set up environment variables
cp config/api_keys_template.txt .env
# Edit .env with your values (see config/api_keys_template.txt)

# 6. Download dataset
kaggle datasets download -d shivamkushwaha/bbc-full-text-document-classification \
    --unzip -p data/raw/

# 7. Start ollama (separate terminal)
ollama pull llama3.2
ollama serve

# 8. Validate setup
python config/settings.py
pytest tests/ -v --tb=short
```

## Google Colab Deployment

The notebooks are designed to run in Google Colab without any local setup:

```python
# First cell in each notebook:
!pip install pyLDAvis scikit-learn vaderSentiment spacy langdetect deep-translator ollama -q
!python -m spacy download en_core_web_sm -q

# For Kaggle data:
from google.colab import files
files.upload()  # upload kaggle.json
!mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d shivamkushwaha/bbc-full-text-document-classification --unzip
```

> Note: ollama cannot run in Colab (no localhost server). For Colab, the LLM cells
> use the Claude API instead (set `ANTHROPIC_API_KEY` in Colab secrets).

## Web Application Deployment

The web application (`NewsBot_IntelligenceSystem_2.html`) is a single self-contained
HTML file that requires no server:

- **In Claude.ai**: Open the file directly — Claude API calls work without a key
- **Local browser**: Open the file, enter your `sk-ant-` API key when prompted
- **GitHub Pages**: Push the HTML file to a `gh-pages` branch and enable Pages

## Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .
EXPOSE 8080

CMD ["python", "-m", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8080", "--no-browser", "--allow-root"]
```

```bash
docker build -t newsbot2 .
docker run -p 8080:8080 newsbot2
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single module
pytest tests/test_preprocessing.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Common Issues

**ImportError on src modules:**
Add the project root to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/ITAI2373-NewsBot-Final:$PYTHONPATH
```

**ollama connection refused:**
```bash
ollama serve   # run this in a separate terminal window
```

**Memory error on large corpus:**
Reduce `TFIDF_MAX_FEATURES` in `config/settings.py` from 5000 to 2000.
