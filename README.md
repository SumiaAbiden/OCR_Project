## Multi-Engine OCR Web Application
DIGILAB OCR is a Streamlit-based web application that provides a unified interface for performing Optical Character Recognition (OCR) using multiple OCR engines. It supports printed and handwritten text, adaptive threshold preprocessing, and AI-based OCR via OpenAI GPT-4o.


**Features:**  
- Upload images in PNG, JPG, or JPEG formats.  
- Supported OCR engines: EasyOCR, EasyOCR + Adaptive Threshold, TrOCR (printed), TrOCR (handwritten), Azure Computer Vision OCR, Tesseract OCR, OpenAI GPT-4o OCR, OpenAI OCR + Feedback. (PaddleOCR is included but currently disabled in the main app — see separate working code file)  
- Visualize recognized text with bounding boxes on the image (where applicable).  
- Provide user feedback to improve OpenAI OCR results.  


**Installation:**  
```bash
# Clone the repository
git clone https://github.com/SumiaAbiden/OCRProject.git
cd OCRProject

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env

#Set keys: `OPENAI_API_KEY` (OpenAI OCR), `AZURE_ENDPOINT` (Azure Computer Vision), `AZURE_SUBSCRIPTION_KEY` (Azure Computer Vision).
```
> Note: PaddleOCR does not require API keys but is temporarily disabled in `app.py`. Use `paddleOCR.py` for PaddleOCR.


**Usage:**  
- Run the app: `streamlit run app.py`  
- Upload an image on the left panel  
- Select the OCR engine on the right panel  
- Click **Run OCR** to extract text
  

**Project Structure:**  
- `app.py` – Main Streamlit application  
- `paddleOCR.py` – Separate PaddleOCR working script  
- `requirements.txt` – Python dependencies  
- `.env.example` – Example environment variables


**Notes:**    
- Ensure a valid OpenAI API key in `.env`.  
- The app automatically detects CUDA/GPU for PyTorch if available.
  

**Supported OCR Methods:**  
- EasyOCR – Standard OCR for printed text  
- EasyOCR + Adaptive Threshold – Improves OCR on low-quality images using adaptive thresholding  
- TrOCR (printed) – Hugging Face TrOCR model for printed text  
- TrOCR (handwritten) – Hugging Face TrOCR model for handwritten text  
- Azure OCR – Microsoft Azure Computer Vision OCR  
- Tesseract OCR – Open-source Tesseract OCR engine  
- OpenAI OCR – GPT-4o-based OCR for challenging images  
- OpenAI OCR + Feedback – Allows user feedback to correct GPT-4o OCR results  
- PaddleOCR – Fast multi-language OCR (disabled in main app, separate working file)

