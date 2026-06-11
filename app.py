import streamlit as st
import tempfile
import os
from agent import load_resume, build_agent

st.set_page_config(
    page_title="AI Job Matching Agent",
    page_icon="🎯",
    layout="centered",
)

st.markdown("""
<style>
    .result-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.header("How to use")
    st.markdown("""
1. Upload your resume as a **PDF**
2. Copy a job description from LinkedIn / Naukri
3. Paste it in the text box
4. Click **Analyze Match**
5. Get your score + skill gap report
    """)
    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("""
- LangChain (LCEL)
- LLaMA3 via Groq
- ChromaDB
- all-MiniLM-L6-v2
- Streamlit
    """)
    st.caption("Built by Atin Choudhary")

# ── HEADER ───────────────────────────────────
st.title("🎯 AI Job Matching Agent")
st.caption("Upload your resume + paste any job description to get a match score and skill gap report.")
st.divider()

# ── INPUTS ───────────────────────────────────
st.markdown("#### Step 1 — Upload your resume")
resume_file = st.file_uploader(
    "Upload PDF", type=["pdf"], label_visibility="collapsed"
)
if resume_file:
    st.success(f"Uploaded: {resume_file.name}")

st.markdown("#### Step 2 — Paste the job description")
job_desc = st.text_area(
    "Job description",
    height=220,
    placeholder="Paste the full job description from LinkedIn, Naukri, etc...",
    label_visibility="collapsed",
)

st.divider()

# ── BUTTON ───────────────────────────────────
analyze_btn = st.button(
    "🔍 Analyze Match",
    type="primary",
    use_container_width=True,
    disabled=(resume_file is None or len(job_desc.strip()) < 50),
)

if not resume_file:
    st.info("Please upload your resume PDF to get started.")
elif len(job_desc.strip()) < 50:
    st.warning("Please paste a full job description (at least 50 characters).")

# ── ANALYSIS ─────────────────────────────────
if analyze_btn and resume_file and len(job_desc.strip()) >= 50:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(resume_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Reading and indexing your resume..."):
            vectorstore = load_resume(tmp_path)

        with st.spinner("Analyzing job description..."):
            chain = build_agent(vectorstore)
            result = chain.invoke(job_desc)

        st.success("Analysis complete!")
        st.divider()
        st.markdown("## Your Match Report")
        st.markdown(result)
        st.divider()

        st.download_button(
            label="📥 Download Report",
            data=result,
            file_name="job_match_report.txt",
            mime="text/plain",
        )

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ── FOOTER ───────────────────────────────────
st.divider()
st.caption("Resume is processed locally and never stored permanently.")