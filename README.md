# Ink2Marks — Question Paper and Answer Sheet Checker

Ink2Marks is a local OCR workflow for scanned exam papers. Upload a question paper and a student's answer sheet, review/correct their OCR text, and request a provisional question-wise marks breakdown with feedback and evidence quotes.

## What it does

- Accepts scanned PDFs and images (PNG, JPG/JPEG, TIFF, BMP) for both question papers and answer sheets.
- Renders PDF pages locally with PyMuPDF.
- Preprocesses page images with OpenCV: grayscale conversion, upscaling of small scans, denoising, adaptive thresholding, and small-angle deskewing.
- Extracts text locally with Tesseract OCR, then lets you correct OCR mistakes in the interface.
- Optionally accepts a marking scheme or model answers to guide marking.
- Can use a configured OpenAI model to propose question-wise marks, feedback, and evidence quotes. It checks that marks are within each question's maximum and that question maxima add to the total requested marks.
- Exports the suggested breakdown to CSV.

**Marks are provisional.** Without a supplied rubric, the model estimates a reasonable rubric from the questions and general subject knowledge; it can be wrong. Review every result before recording a grade. Handwriting, OCR quality, ambiguous questions, and subject-specific marking rules can affect results.

## Privacy

OCR runs locally. The optional AI grading step sends OCR text from both uploaded papers and any rubric entered in the app to the OpenAI API using your configured API key. It is off until you tick the consent box and click **Grade answer sheet**. Use it only if you are permitted to share that student work with the configured provider. No API key is required for OCR, but a key is required to grade automatically.

## Install

You need Python 3.10–3.12, Tesseract OCR 5.x, and the Python packages below. The Tesseract executable is a separate system install; installing `pytesseract` alone is not enough.

### Windows PowerShell

1. Install Tesseract OCR for Windows and include the language data you need (for example, English `eng`).
2. In this project folder, run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

3. Add your API key to `.env` to enable AI grading, and optionally set the executable path if Tesseract is not on PATH:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

4. Start the app:

```powershell
streamlit run app.py
```

If you only need OCR and do not want AI grading, leave `OPENAI_API_KEY` blank. The app can extract/edit/download text; automatic marks are not available without an AI key.

### macOS/Linux

Install Tesseract through your package manager (`brew install tesseract` on macOS or `sudo apt install tesseract-ocr` on Ubuntu/Debian). Then create a virtual environment, run `pip install -r requirements.txt`, and launch with `streamlit run app.py`.

## Use

1. Upload both the **question paper** and **student answer sheet**.
2. Wait for OCR, then correct any recognition errors in both text boxes.
3. Optionally paste the official marking scheme or reference answers.
4. Set the exam's total marks.
5. Review the privacy note. Tick the consent box and click **Grade answer sheet** to send the OCR text to your configured OpenAI model.
6. Review suggested marks, per-question feedback, and evidence quotes; download the CSV if useful.

For multilingual papers, install the matching Tesseract language data and set the language code in the sidebar (for example `eng+hin`).

## Limitations

- The app does not independently verify official answer correctness. AI-suggested marks need a teacher's review.
- No reference answer or marking rubric means the model must infer grading criteria from the question. Provide the official rubric whenever possible.
- OCR may misread handwriting, diagrams, equations, tables, or low-quality scans. Correct the editable OCR text before grading.
- Automatic alignment of subparts is model-based and may need correction.
- Local OCR does not require an API key. AI marking requires a user-configured OpenAI API key, internet access, and a model available to that key.
