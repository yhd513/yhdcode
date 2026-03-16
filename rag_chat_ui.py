import streamlit as st
import yhd

yhd.rag_init()  # 初始化 rag 环境

# ----------------------
# 页面配置
# ----------------------
st.set_page_config(
    page_title="RAG 智能问答",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG 检索问答系统")

# ----------------------
# 初始化聊天历史
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------
# 显示历史对话
# ----------------------
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    with st.chat_message(role):
        st.markdown(content)

# ----------------------
# 输入框
# ----------------------
prompt = st.chat_input("请输入你的问题...")

if prompt:
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 回答（这里对接你的 RAG 核心函数）
    with st.chat_message("assistant"):
        with st.spinner("检索中..."):

            # ======================
            # 【在这里替换成你的 RAG 函数】
            # ======================
            def rag_answer(question):
                """你的 RAG 核心逻辑：检索 + 大模型回答"""
                result = yhd.rag_answer(question)

                return f"{result}\n"
                # return f"【RAG 检索结果】："

            # 调用 RAG
            response = rag_answer(prompt)

            # 显示回答
            st.markdown(response)

    # 3. 保存到历史
    st.session_state.messages.append({"role": "assistant", "content": response})