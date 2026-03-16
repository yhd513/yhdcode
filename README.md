# 运行方法

## 安装依赖

然后在项目根目录下创建一个名为 `.env` 的文件，并添加以下内容：

```env
GEMINI_API_KEY=AIzaSyChgeTXMCS3_44zeMePCrd8GZV-OPsvVlk
```

其中 xxx 为你的 Google Gemini API 密钥。没有密钥的用户可以在 https://aistudio.google.com/apikey 上申请。

## 使用模型

向量计算：本地欧拉（ollama.rnd.huawei.com/library/bge-m3:latest） 可以选择其他的

重排算法：sentence_transformers import CrossEncoder（cross-encoder/mmarco-mMiniLMv2-L12-H384-v1）

大语言模型：openai(qwen2.5-72b-instruct)

## 运行
启动前端页面：streamlit run rag_chat_ui.py
对话地址：http://localhost:8501/

