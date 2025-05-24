
import streamlit as st
import pandas as pd
from pandasai.smart_dataframe import SmartDataframe
from pandasai.llm.openai import OpenAI
from vanna.remote import VannaDefault
st.set_page_config(page_title="Automated Analysis", layout="centered", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stTextInput>div>div>input {
            background-color: #1f1f28;
            color: white;
        }
        .stDataFrame, .stMarkdown {
            color: #dddddd;
        }
        h1, h2, h3 {
            color: #00d4d4;
        }
        .css-1cpxqw2 { background-color: #1f1f28; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Automated Data Analysis Assistant")
engine = st.sidebar.radio("🧠 Choose Analysis Engine", ("PandasAI", "Vanna"))
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("### 📊 Data Preview (Top 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)

    user_question = st.text_input("💬 Ask a question about your uploaded data:")

    if engine == "PandasAI":
        @st.cache_resource
        def init_smart_df(dataframe):
            llm = OpenAI(api_token="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")
            return SmartDataframe(dataframe.head(30), config={"llm": llm, "enable_cache": False})

        smart_df = init_smart_df(df)

        if user_question:
            with st.spinner("⚡ PandasAI is analyzing..."):
                try:
                    response = smart_df.chat(user_question)
                    st.success("Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif engine == "Vanna":
        @st.cache_resource
        def init_vanna():
            vn = VannaDefault(model='chinook', api_key='6c81999aa29147e59aad7e307a2f6503')
            vn.connect_to_sqlite('https://vanna.ai/Chinook.sqlite')
            return vn

        vanna = init_vanna()

        if user_question:
            with st.spinner("Vanna is analyzing..."):
                try:
                    response = vanna.ask(user_question, visualize=False)
                    st.success(" Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("📤 Please upload a CSV file to begin.")
