import streamlit as st
from rag import yhd

yhd.rag_init()  # 初始化 rag 环境

# ----------------------
# 页面配置
# ----------------------
st.set_page_config(
    page_title="RAG 智能问答",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG 智能问答")

# ----------------------
# 初始化聊天历史（上下文记忆）
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
    # ======================
    # 修复：用户消息只添加一次
    # ======================
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 回答（对接你的 RAG）
    with st.chat_message("assistant"):
        with st.spinner("检索中..."):

            def rag_answer(question):
                """RAG 核心逻辑 + 上下文记忆传入"""
                # ======================
                # ✅ 关键：把历史对话传给 RAG（让AI拥有上下文）
                # ======================
                result = yhd.rag_answer(question, st.session_state.messages)
                return f"{result}"

            # 调用 RAG
            response = rag_answer(prompt)

            # 显示回答
            st.markdown(response)

    # 3. 把AI回答存入记忆（上下文核心）
    st.session_state.messages.append({"role": "assistant", "content": response})