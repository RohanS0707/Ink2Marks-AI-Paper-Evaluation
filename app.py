"""Ink2Marks: OCR question papers and answer sheets, then review marks."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ink2marks.grading import GradingError, grade_exam
from ink2marks.ocr import extract_text, load_pages

load_dotenv()
st.set_page_config(page_title="Ink2Marks - Answer Sheet Checker", page_icon="📝", layout="wide")
st.title("📝 Ink2Marks — Answer Sheet Checker")
st.caption("Upload a question paper and a student's answer sheet. OCR both, then get a provisional question-wise marks breakdown.")
st.warning("AI marks are suggestions for teacher review, not official grades. OCR errors, subject context, and grading criteria can affect the result.")

with st.sidebar:
    st.header("OCR settings")
    language = st.text_input("Tesseract language code", value="eng", help="Examples: eng or eng+hin. Install language data in Tesseract first.")
    use_preprocessing = st.toggle("Use OpenCV preprocessing", value=True)
    st.caption("OpenCV applies grayscale conversion, denoising, adaptive thresholding, and small-angle deskewing.")

left, right = st.columns(2)
with left:
    question_file = st.file_uploader("1. Question paper", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="question_file")
with right:
    answer_file = st.file_uploader("2. Student answer sheet", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="answer_file")

if not question_file or not answer_file:
    st.info("Upload both documents to continue. PDF pages and image files are supported.")
    st.stop()

signature = f"{question_file.name}:{question_file.size}|{answer_file.name}:{answer_file.size}|{language}|{use_preprocessing}"
if st.session_state.get("exam_signature") != signature:
    try:
        with st.spinner("Rendering documents and extracting text with OCR..."):
            extracted = {}
            for doc_key, uploaded in (("question", question_file), ("answer", answer_file)):
                pages = load_pages(uploaded.name, uploaded.getvalue())
                page_texts = []
                for page_number, page in enumerate(pages, 1):
                    text, processed = extract_text(page, language=language, preprocess=use_preprocessing)
                    page_texts.append(f"--- Page {page_number} ---\n{text}")
                extracted[doc_key] = "\n\n".join(page_texts)
        st.session_state["question_ocr"] = extracted["question"]
        st.session_state["answer_ocr"] = extracted["answer"]
        st.session_state["exam_signature"] = signature
        st.session_state.pop("grade_result", None)
    except Exception as exc:
        st.error(f"OCR could not process the documents: {exc}")
        st.stop()

st.subheader("Review OCR text")
st.caption("Correct OCR mistakes before grading. Printed and clear scans usually work better than handwriting.")
colq, cola = st.columns(2)
with colq:
    question_text = st.text_area("Question paper text", value=st.session_state["question_ocr"], height=300, key="question_text_edit")
with cola:
    answer_text = st.text_area("Student answer text", value=st.session_state["answer_ocr"], height=300, key="answer_text_edit")

with st.expander("Optional marking scheme / reference answers", expanded=False):
    marking_scheme = st.text_area("Paste rubric, expected points, or model answers", height=160, key="marking_scheme")

settings_col, action_col = st.columns([1, 2])
with settings_col:
    total_marks = st.number_input("Total marks", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    model_name = st.text_input("Grading model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
with action_col:
    st.markdown("**Privacy and grading**")
    st.caption("AI grading sends the OCR text of both papers, plus any rubric you provide, to the OpenAI API configured by your API key. The key is read from `.env`. Keep this disabled unless you are allowed to share the student work with that provider.")
    consent = st.checkbox("I understand and approve sending these OCR texts to the configured OpenAI model for provisional grading.")
    grade_clicked = st.button("Grade answer sheet", type="primary", disabled=not consent, use_container_width=True)

if grade_clicked:
    try:
        with st.spinner("Assessing answers and preparing a question-wise breakdown..."):
            st.session_state["grade_result"] = grade_exam(
                question_text=question_text,
                answer_text=answer_text,
                total_marks=total_marks,
                marking_scheme=marking_scheme if "marking_scheme" in locals() else "",
                model=model_name.strip() or None,
            )
    except GradingError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Grading failed: {exc}")

if st.session_state.get("grade_result"):
    result = st.session_state["grade_result"]
    st.subheader("Provisional result")
    score_col, model_col = st.columns([1, 2])
    with score_col:
        st.metric("Suggested marks", f"{result['awarded']:g} / {result['maximum']:g}")
    with model_col:
        st.caption(f"Generated with {result['model']}. Review every score and evidence quote before recording a grade.")

    table = pd.DataFrame([
        {"Question": row["question"], "Marks": f"{row['marks_awarded']:g} / {row['max_marks']:g}", "Feedback": row["feedback"]}
        for row in result["questions"]
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    for row in result["questions"]:
        with st.expander(f"{row['question']} — {row['marks_awarded']:g}/{row['max_marks']:g}"):
            st.write(row["feedback"])
            st.markdown("**Evidence quote from answer sheet**")
            st.write(row["evidence"] or "No evidence quote supplied.")

    csv = pd.DataFrame(result["questions"]).to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download marks breakdown (CSV)", data=csv, file_name="ink2marks_results.csv", mime="text/csv")
