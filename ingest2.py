import os
import warnings
# 屏蔽所有无关警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 稳定兼容导入，锁定0.1.x langchain版本专用
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 读取PDF文件夹配置
docs_dir = "docs"
all_documents = []

# 自动创建docs文件夹，防止找不到目录报错
if not os.path.exists(docs_dir):
    os.makedirs(docs_dir)
    print(f"已自动创建【{docs_dir}】文件夹，请把PDF文件放进这个文件夹后重新运行！")
    exit()

# 遍历读取所有PDF
for filename in os.listdir(docs_dir):
    if filename.lower().endswith(".pdf"):
        full_path = os.path.join(docs_dir, filename)
        loader = PyMuPDFLoader(full_path)
        page_docs = loader.load()
        all_documents.extend(page_docs)
        print(f"成功加载文件：{filename}，共 {len(page_docs)} 页")

# 判断是否存在PDF文件
if len(all_documents) == 0:
    print("警告：docs文件夹内没有检测到任何PDF文件，程序退出！")
    exit()

# 文本分割器配置
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
text_chunks = text_splitter.split_documents(all_documents)
print(f"\n文本分割完成，一共生成 {len(text_chunks)} 个文本片段")

# 加载向量化模型
print("\n正在加载多语言嵌入模型，首次运行会自动下载模型文件，请耐心等待...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 构建并持久化向量库
vector_db = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)
vector_db.persist()

print("\n==================== 执行成功 ====================")
print("PDF知识库构建完成！向量数据已保存在 chroma_db 文件夹中")