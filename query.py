import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# ── 1. 加载向量数据库 ─────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# ── 2. 连接 DeepSeek API ──────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ── 3. 问答函数 ───────────────────────────────────────────────
def ask(question):
    results = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""你是一个文档助手，请根据以下内容回答用户的问题。
如果内容中没有相关信息，请说"文档中未找到相关内容"。

参考内容：
{context}

用户问题：{question}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ── 4. 循环问答 ───────────────────────────────────────────────
print("📚 RAG 问答系统已启动，输入 exit 退出\n")
while True:
    question = input("你的问题：")
    if question.strip() == "exit":
        break
    answer = ask(question)
    print(f"\n回答：{answer}\n")
