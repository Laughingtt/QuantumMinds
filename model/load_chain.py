from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from .DocSplit import get_files, get_text
from BCEmbedding.tools.langchain import BCERerank
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .intern_llm import InternLM_LLM

MODEL_PATH = "/mnt/workspace/llm/model/merged3"
KNOWLEDGE_DB_PATH = "/mnt/workspace/llm/knowledge_db"
EMBEDDING_MODEL_NAME = '/mnt/workspace/llm/bce/bce-embedding-base_v1'
RERANKER_MODEL = '/mnt/workspace/llm/bce/bce-reranker-base_v1'


# 直接调用llm模型
def load_chain_single():
    # 加载自定义 LLM
    llm = InternLM_LLM(model_path=MODEL_PATH)
    return llm.predict


# 增加向量知识库
def load_chain(qa_mode=1):
    # 切分文件
    # 目标文件夹
    tar_dir = [KNOWLEDGE_DB_PATH]
    # tar_dir = ["/mnt/workspace/llm/kd_tmp/"]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # 加载问答链
    # init embedding model
    embedding_model_name = EMBEDDING_MODEL_NAME
    embedding_model_kwargs = {'device': 'cuda:0'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )
    # 加载重排序模型
    reranker_args = {'model': RERANKER_MODEL, 'top_n': 1, 'device': 'cuda:0'}
    reranker = BCERerank(**reranker_args)

    retriever = FAISS.from_documents(split_docs, embed_model,
                                     distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
        search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 5})

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )

    # 加载自定义 LLM
    llm = InternLM_LLM(model_path=MODEL_PATH)

    if qa_mode == 1:
        """
        1. 检索QA链,创建检索QA链，用于合并检索到的文本片段并调用语言模型
        """
        # 定义一个 Prompt Template
        template = """请参考以下上下文来回答最后的问题。如果上下文的内容不相关，请按照你自己的理解，尽量用温和的语气鼓励，安慰提问者。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        有用的回答:"""

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        # 运行 chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever, return_source_documents=True,
                                               chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    elif qa_mode == 2:
        """
        2. 对话检索链,传入语言模型、检索器和记忆系统
        """
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=compression_retriever,
            memory=memory
        )
    else:
        qa_chain = None

    return qa_chain
