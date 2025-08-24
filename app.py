import os
import streamlit as st
from dotenv import load_dotenv
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
import torch
from tqdm.auto import tqdm
import easyocr
import numpy as np
import base64
from paddleocr import PaddleOCR
import tempfile
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import subprocess
import io
from openai import OpenAI


load_dotenv()


# ---------- Helper Classes ----------

class Base64ImageHelper:
    @staticmethod
    def get_base64_image(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()


class ImagePreprocessor:
    @staticmethod
    def resize_with_padding(image, target_size=(384, 384)):
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        delta_w = target_size[0] - image.width
        delta_h = target_size[1] - image.height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(image, padding)


# ---------- OCR Model Classes ----------

class EasyOCRWrapper:
    """
        Wrapper class for performing Optical Character Recognition (OCR) using the EasyOCR library.

        This class initializes an EasyOCR reader for specified languages and provides methods
        to extract text from images. It supports both direct image reading and adaptive thresholding
        preprocessing to improve OCR accuracy on challenging images.

        Change language settings or other parameters as needed.
        """
    def __init__(self, langs=['en']):
        self.reader = easyocr.Reader(langs)

    from PIL import Image, ImageDraw, ImageFont

    def infer(self, image_path_or_file):
        if not isinstance(image_path_or_file, str):
            image = Image.open(image_path_or_file).convert("RGB")
        else:
            image = Image.open(image_path_or_file).convert("RGB")

        results = self.reader.readtext(np.array(image))

        text_output = "\n".join([res[1] for res in results])

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except:
            font = ImageFont.load_default()

        for (bbox, text, prob) in results:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            draw.rectangle([top_left, bottom_right], outline="green", width=2)
            draw.text((top_left[0], top_left[1] - 15), text, fill="green", font=font)

        return text_output, image

    def infer_adaptive_threshold(self, uploaded_file):
        # Görseli aç ve griye çevir
        image = Image.open(uploaded_file).convert("L")

        # Adaptive Threshold uygulanmış ikili görsel elde et
        image_np = np.array(image)
        mean = Image.fromarray(image_np).filter(ImageFilter.GaussianBlur(1))
        image_bin = (image_np < mean).astype(np.uint8) * 255
        thresh_img = Image.fromarray(image_bin).convert("RGB")

        # OCR işlemi (EasyOCR, numpy array bekliyorsa)
        results = self.reader.readtext(np.array(thresh_img))

        # Sonuçları çizmek için renkli görsel
        draw = ImageDraw.Draw(thresh_img)
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except:
            font = ImageFont.load_default()

        for (bbox, text, prob) in results:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            draw.rectangle([top_left, bottom_right], outline="green", width=2)
            draw.text((top_left[0], top_left[1] - 15), text, fill="green", font=font)

        text_output = "\n".join([text for (_, text, _) in results])
        return text_output, thresh_img


class TrOCRPrintedModel:
    def __init__(self, model_name="microsoft/trocr-base-printed"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer(self, image):
        image = image.convert("RGB").resize((384, 384))
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=256)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text



class TrOCRHandwritten:
    """
        Specialized TrOCR model wrapper for handwritten text recognition.

        Inherits from TrOCRModel but uses the pretrained "microsoft/trocr-base-handwritten" model

        Provides a simple infer method for predicting IAM handwritten text from a given image.
        """

    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        # Sadece pretrained yolla — EKSTRA PARAMETRE YOK
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)  # NO device_map!

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Bu artık güvenli

    def infer(self, image):
        image = image.convert("RGB").resize((384, 384))
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=256)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text


class PaddleOCRWrapper:
    """
        Wrapper class for performing OCR using PaddleOCR.

        This class initializes the PaddleOCR engine with specific options disabling
        document orientation classification and text line orientation for faster
        processing. It reads an uploaded image file, performs OCR, and annotates
        detected text regions on the image.

        Returns recognized text and a PIL image with bounding boxes and text overlays.
        """
    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )


    def infer(self, uploaded_file):
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_image_path = tmp_file.name

        result = self.ocr.predict(temp_image_path)

        image = Image.open(temp_image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except:
            font = ImageFont.load_default()

        # Metin kutularını ve yazıları çiz
        for line in result:
            bbox = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text = line[1][0]

            polygon = [tuple(point) for point in bbox]
            draw.polygon(polygon, outline="green")
            draw.text((bbox[0][0], bbox[0][1] - 15), text, fill="green", font=font)

        # Metinleri birleştir
        text_lines = [line[1][0] for line in result]
        return "\n".join(text_lines), image


class AzureOCRWrapper:
    """
        Wrapper for performing OCR using Microsoft's Azure Computer Vision API.

        This class initializes the Azure Computer Vision client with credentials from environment variables,
        sends an image for OCR processing, and polls the service until the operation completes.

        Returns extracted text as a concatenated string from all recognized lines.
    """
    def __init__(self):
        self.client = ComputerVisionClient(
            os.getenv("AZURE_ENDPOINT"),
            CognitiveServicesCredentials(os.getenv("AZURE_SUBSCRIPTION_KEY"))
        )

    def infer(self, uploaded_file):
        img_bytes = uploaded_file.read()
        image_stream = io.BytesIO(img_bytes)
        read_response = self.client.read_in_stream(image_stream, raw=True)

        operation_location = read_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        while True:
            result = self.client.get_read_result(operation_id)
            if result.status not in ['notStarted', 'running']:
                break

        if result.status == OperationStatusCodes.succeeded:
            text_lines = []
            for page in result.analyze_result.read_results:
                for line in page.lines:
                    text_lines.append(line.text)
            return "\n".join(text_lines)
        else:
            raise Exception("Azure OCR failed or timed out.")


class TesseractOCRWrapper:
    """
        Wrapper class to perform OCR using the Tesseract OCR engine via command line.

        Initializes with the Tesseract executable path and provides a method to
        perform OCR on uploaded image files by saving them temporarily and
        running the Tesseract CLI. Supports specifying the language for OCR.

        Returns the recognized text or None if an error occurs.
        """
    def __init__(self, tesseract_path="tesseract"):
        self.tesseract_path = tesseract_path

    def infer(self, uploaded_file, lang='eng'):
        uploaded_file.seek(0)
        fd, tmp_img_path = tempfile.mkstemp(suffix=".png")
        with os.fdopen(fd, 'wb') as tmp_img_file:
            tmp_img_file.write(uploaded_file.read())

        tmp_output_base = tmp_img_path + "_out"

        try:
            cmd = [self.tesseract_path, tmp_img_path, tmp_output_base, '-l', lang]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(tmp_output_base + ".txt", "r", encoding="utf-8") as f:
                result_text = f.read()
        except subprocess.CalledProcessError:
            result_text = None
        finally:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
            if os.path.exists(tmp_output_base + ".txt"):
                os.remove(tmp_output_base + ".txt")

        return result_text


class OpenAIOCRWrapper:
    """
        Wrapper class for performing OCR using OpenAI's GPT-4o model with image input.

        Initializes the OpenAI client with an API key and sends images encoded in base64
        as part of a chat completion request. The system prompt instructs the model to act
        as an OCR expert and handle edge cases carefully.

        Returns the recognized text extracted from the image or a specific message if no text is detected.
    """
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)

    def infer(self, image):

        image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        response = self.client.chat.completions.create( # ✅ DOĞRU

            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR expert. If no text is readable, respond with exactly: No text detected. Be"
                        "careful for 0 and o or splash chracters for example PL3758-3 must be. Write only the exact text"
                        "without any additions"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64.b64encode(img_bytes).decode()
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            # temperature=1  # Deterministic yapm ak için 0 yap (Aynı sonucu verme olasılığı artar. )
        )
        print("Response comes ... ")
        return response.choices[0].message.content

    def infer_with_feedback(self, image, user_feedback: str = None):
        image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        #OCR ile metni çıkarma
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR expert. If no text is readable, respond with exactly: No text detected. "
                        "Be careful for 0 and o or splash characters. For example, PL3758-3 must be. "
                        "Write only the exact text without any additions."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64.b64encode(img_bytes).decode()
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
        )

        ocr_text = response.choices[0].message.content.strip()

        #Kullanıcı feedback'i varsa GPT ile düzeltme
        if user_feedback and ocr_text != "No text detected":
            correction_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant helping improve OCR output using user feedback. "
                            "You will receive an OCR result and a user's comment describing what is wrong. "
                            "Based on that, output a corrected version of the OCR result. Return ONLY the corrected text."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"OCR result:\n{ocr_text}\n\n"
                            f"User feedback:\n{user_feedback}\n\n"
                            "Correct the OCR result accordingly."
                        )
                    }
                ],
                max_tokens=1000,
            )

            corrected_text = correction_response.choices[0].message.content.strip()
        else:
            corrected_text = ocr_text

        return ocr_text, corrected_text


# ---------- Streamlit UI ----------

class OCRApp:
    """
       Main application class integrating multiple OCR engines into a single Streamlit UI.

       This class initializes wrappers for various OCR models (EasyOCR, TrOCR, PaddleOCR, Azure OCR, Tesseract, OpenAI OCR),
       sets up the web UI with file upload, method selection, and displays OCR results with progress feedback.

       It provides a unified interface to run OCR inference using different backends and visualize both input images and results.
       """
    def __init__(self):
        self.easyocr = EasyOCRWrapper()
        self.trocr_printed = TrOCRPrintedModel()
        self.trocr_handwritten = TrOCRHandwritten()
        self.paddleocr = PaddleOCRWrapper()
        self.azureocr = AzureOCRWrapper()
        self.tesseract = TesseractOCRWrapper()
        self.openaiocr = OpenAIOCRWrapper()
        st.set_page_config(page_title="DIGILAB OCR", layout="centered")

    def run(self):
        self._render_ui()


    def _render_ui(self):
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.header("📤 Upload Image")
                uploaded_file = st.file_uploader("Add your image", type=["png", "jpg", "jpeg"])
                if uploaded_file:
                    if st.session_state.get("last_uploaded") != uploaded_file.name:
                        st.session_state["ocr_text"] = ""
                        st.session_state["last_uploaded"] = uploaded_file.name
                    st.image(uploaded_file, caption="Preview", use_container_width=True)

            with col2:
                st.header("📝 OCR Result")
                if uploaded_file:
                    method = st.radio("Select Inference Method", [
                        "microsoft/trocr-base-printed",
                        "EasyOCR",
                        "EasyOCR + Adaptive Threshold",
                        "microsoft/trocr-base-handwritten",
                        "PaddleOCR",
                        "Azure OCR",
                        "Tesseract OCR",
                        "OpenAI OCR",
                        "OpenAI OCR + Feedback"
                    ])

                    disable_run = (method == "PaddleOCR")

                    if disable_run:
                        st.warning("PaddleOCR disabled right now. Please Try another model.")

                    if st.button("🚀 Run OCR"):
                        progress_bar = st.progress(0)
                        with st.spinner("Running OCR..."):
                            try:
                                if method == "microsoft/trocr-base-printed":
                                    image = Image.open(uploaded_file)
                                    text = self.trocr_printed.infer(image)
                                elif method == "EasyOCR":
                                    text, vis_img = self.easyocr.infer(uploaded_file)
                                    st.image(vis_img, caption="EasyOCR Output", use_container_width=True)
                                elif method == "EasyOCR + Adaptive Threshold":
                                    text, vis_img = self.easyocr.infer_adaptive_threshold(uploaded_file)
                                    st.image(vis_img, caption="EasyOCR + Adaptive Threshold Output",
                                             use_container_width=True)
                                elif method == "microsoft/trocr-base-handwritten":
                                    image = Image.open(uploaded_file)
                                    text = self.trocr_handwritten.infer(image)
                                elif method == "PaddleOCR":
                                    text, vis_img = self.paddleocr.infer(uploaded_file)
                                    st.image(vis_img, caption="PaddleOCR Output", use_container_width=True)
                                elif method == "Azure OCR":
                                    text = self.azureocr.infer(uploaded_file)
                                elif method == "Tesseract OCR":
                                    text = self.tesseract.infer(uploaded_file)
                                elif method == "OpenAI OCR":
                                    image = Image.open(uploaded_file)
                                    text = self.openaiocr.infer(image)
                                elif method == "OpenAI OCR + Feedback":
                                    image = Image.open(uploaded_file)
                                    text, feedback = self.openaiocr.infer_with_feedback(image)
                                    st.text_area("Feedback on OCR Result:", feedback, height=200, disabled=True)
                                else:
                                    text = ""
                                st.session_state["ocr_text"] = text
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")
                                st.session_state["ocr_text"] = ""
                        progress_bar.empty()

                    if method == "OpenAI OCR + Feedback" and uploaded_file:
                        st.text_area("OCR Result (Initial):", st.session_state.get("ocr_text", ""), height=150,
                                     disabled=True)
                        user_feedback = st.text_area("Provide feedback or corrections here:",
                                                     value=st.session_state.get("user_feedback", ""),
                                                     key="user_feedback_input")

                    if st.button("Submit Feedback"):
                        feedback = user_feedback.strip()
                        if feedback == "":
                            st.warning("Please provide some feedback before submitting.")
                        else:
                            image = Image.open(uploaded_file)
                            _, corrected_text = self.openaiocr.infer_with_feedback(image, user_feedback=feedback)
                            st.session_state["corrected_text"] = corrected_text
                            st.session_state["user_feedback"] = feedback
                            st.success("Corrected Text Based on Feedback:")
                            st.text_area("Corrected OCR Text:", corrected_text, height=150, disabled=True)
                    else:
                        if st.session_state.get("corrected_text"):
                            st.text_area("Corrected OCR Text (previous):", st.session_state["corrected_text"],
                                         height=150, disabled=True)

                if "ocr_text" in st.session_state and st.session_state["ocr_text"]:
                    st.success("✅ OCR Done Successfully!!")
                    st.text_area("Recognized Text:", st.session_state["ocr_text"], height=300, disabled=True)
                elif not uploaded_file:
                    st.info("Please upload an image to begin.")
                else:
                    st.info("Please run OCR to see results.")


if __name__ == "__main__":
    app = OCRApp()
    app.run()
