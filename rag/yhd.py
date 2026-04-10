from typing import List
import requests
import chromadb
from sentence_transformers import CrossEncoder
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"    # 关闭警告


# Ollama 本地接口地址（固定不用改）
OLLAMA_API = "http://localhost:11434/api/embed"
EMBED_MODEL = "ollama.rnd.huawei.com/library/bge-m3:latest"
llm_model_id_local="ollama.rnd.huawei.com/library/qwen2.5:latest"
llm_base_url_local="http://localhost:11434/api/chat"

llm_model_id="qwen2.5-72b-instruct"
llm_api_key="Bearer sk-1234"
llm_base_url="http://api.openai.rnd.huawei.com/v1/chat/completions"
chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection("default")
use_local = True

cross_encoder_key='D:\\工作\\工作\\AI_Agent\\code\\ragyhd\\ai\\model\\all-MiniLM-L6-v2'




# ===================== 分片 =====================
def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r', encoding="utf-8") as file:
        content = file.read()

    return [chunk for chunk in content.split("\n\n")]


# ===================== 调用本地 Ollama 生成向量 =====================
def embed_chunk(chunk: str) -> List[float]:
    payload = {
        "model": EMBED_MODEL,
        "input": chunk
    }
    response = requests.post(OLLAMA_API, json=payload)
    return response.json()["embeddings"][0]

def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )

def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

def reorder(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder(cross_encoder_key)
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    chunk_with_scores_list = [(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]
    chunk_with_scores_list.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in chunk_with_scores_list][:top_k]

def generate(query: str, history: List[str], chunks: List[str]) -> str:
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{"\n\n".join(chunks)}

历史消息: {history}

请基于上述内容回答。"""

    print(f"{prompt}\n\n-------------------------------------\n")

    print(f"回答:\n")
    return call_model(prompt)

def call_model(prompt: str) -> str:
    headers = {
        'Authorization': llm_api_key,
        'Content-Type': 'application/json'
    }

    if use_local:
        data = {
            'model': llm_model_id_local,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            # 本地
            "stream": False
        }
    else:
        data = {
            'model': llm_model_id_local,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
        }

    response = requests.post(llm_base_url_local, headers=headers, json=data)
    if use_local:
        # 本地
        answer = response.json().get('message').get('content')
    else:
        # openAI
        answer = response.json()["choices"][0].get('message').get('content')
    print(answer)
    return answer


# ===================== 测试 =====================
if __name__ == "__main__":
    ## 提问前

    #1、分片
    chunks = split_into_chunks("../knowledagebase/full_service_area.md")

    #2、生成向量
    embeddings = [embed_chunk(chunk) for chunk in chunks]

    #3、保存向量
    chromadb_client = chromadb.EphemeralClient()
    chromadb_collection = chromadb_client.get_or_create_collection("default")
    save_embeddings(chunks, embeddings)

    ## 提问后
    query = "综合业务区规划App有哪些计算参数？"

    #4、召回
    retrieved_chunks = retrieve(query, 5)

    #5、重排
    reordered_chunks = reorder(query, retrieved_chunks, 3)

    #6、生成
    generate(query, '', retrieved_chunks)

def rag_init():
    ## 提问前

    #1、分片
    chunks = split_into_chunks("../knowledagebase/full_service_area.md")

    #2、生成向量
    embeddings = [embed_chunk(chunk) for chunk in chunks]

    #3、保存向量
    save_embeddings(chunks, embeddings)


def rag_answer(question: str, history: List[str]) -> str:
    #4、召回
    retrieved_chunks = retrieve(question, 5)

    #5、重排
    # reordered_chunks = reorder(question, retrieved_chunks, 2)

    #6、生成
    return generate(question, history, retrieved_chunks)