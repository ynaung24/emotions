import streamlit as st
from evaluate import evaluate_response, load_corpus

st.set_page_config(page_title="Interview Evaluator", layout="centered")
st.title("💬 Interview Response Evaluator")

# Load corpus and get questions
corpus = load_corpus()
questions = [q["text"] for q in corpus["questions"]]

# UI Elements
question = st.selectbox("Choose a question:", questions)
user_answer = st.text_area("Your response:")

if st.button("Evaluate"):
    if not user_answer.strip():
        st.warning("Please enter a response first.")
    else:
        evaluation = evaluate_response(user_answer, question)
        
        # Display results
        st.subheader(f"🧠 Overall Score: {evaluation['score']}/100")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Relevance", f"{evaluation['metrics']['relevance']}/100")
        with col2:
            st.metric("Clarity", f"{evaluation['metrics']['clarity']}/100")
        with col3:
            st.metric("Completeness", f"{evaluation['metrics']['completeness']}/100")
        
        # Display feedback
        st.info(f"📋 Feedback: {evaluation['feedback']}")