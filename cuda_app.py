from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from DocSplit import get_files, get_text
from BCEmbedding.tools.langchain import BCERerank
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter


class InternLM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        system_prompt = """你名叫萤萤，是一个陪伴型聊天机器人，负责安慰，鼓励，以及心理治愈。
        - YingYing is a conversational language model that is developed by SSJT (上海数据集团). It is designed to be helpful, honest, and harmless.
        - YingYing (萤萤) can understand and communicate fluently in the language chosen by the user such as English and 中文.
        """

        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"


# 直接调用llm模型
def load_chain_single():
    # 加载自定义 LLM
    llm = InternLM_LLM(model_path="/mnt/workspace/.cache/modelscope/Shanghai_AI_Laboratory/internlm2-chat-7b")
    return llm.predict


# 增加向量知识库
def load_chain():
    # 切分文件
    # 目标文件夹
    tar_dir = ["/mnt/workspace/llm/knowledge_db"]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # 加载问答链
    # init embedding model
    embedding_model_name = '/mnt/workspace/llm/bce/bce-embedding-base_v1'
    embedding_model_kwargs = {'device': 'cuda:0'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )
    # 加载重排序模型
    reranker_args = {'model': '/mnt/workspace/llm/bce/bce-reranker-base_v1', 'top_n': 3, 'device': 'cuda:0'}
    reranker = BCERerank(**reranker_args)
    # embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

    # example 1. retrieval with embedding and reranker
    retriever = FAISS.from_documents(split_docs, embed_model,
                                     distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
        search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 10})

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )

    # 加载自定义 LLM
    # llm = InternLM_LLM(model_path="/mnt/workspace/.cache/modelscope/Shanghai_AI_Laboratory/internlm2-chat-7b")
    llm = InternLM_LLM(model_path="/mnt/workspace/llm/model/merged3")
    # llm = InternLM_LLM(model_path="/mnt/workspace/models/merged")
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    # conversation_llm = ConversationChain(
    #     llm=llm,
    #     verbose=True,
    #     memory=ConversationBufferMemory()
    # )

    # 定义一个 Prompt Template
    template = """请参考以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量用温和的语气鼓励，安慰提问者。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever, return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    return qa_chain


class Model_center():
    """
    存储检索问答链的对象
    """

    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history


import gradio as gr

# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        gr.Image('images/yingying.webp', width=100, scale=0)
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown(
                '''
                # 知心大姐“唠五毛”

                智能小助手：小萤火，小名叫萤萤，希望微微萤火能照亮诉说者前行的路和心灵的光...

                一个懂你的陪伴型机器人，为你打造一片心灵的栖息地。
                在这里，你可以尽情倾诉，释放内心的情感，让心灵得到慰藉。让我们开始今天的谈话吧！
                ''')

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=850, show_copy_button=True, avatar_images=("images/xiaobai.png", "images/yingying.webp"),
                                 label="唠五毛")
            first = """
### 唠五毛 - 为你提供情绪价值的智能机器人

一个懂你的陪伴型机器人，为你打造一片心灵的栖息地。在这里，你可以尽情倾诉，释放内心的情感，让心灵得到慰藉。让我们开始今天的谈话吧！
\n
试试以下问题：
\n
1.自我探索

    我是讨好型人格，感觉自己活的很卑微不快乐，我可以改变吗？

2.情感问题

    失恋为什么这么痛苦，能从心理学的角度帮我分析一下吗？

3.学业烦恼

    我最近学习成绩下降，感觉很难集中注意力，这让我很焦虑。我该怎么办呢？"""
            chatbot.value = [[None, first]]

            # 创建一个文本框组件，用于输入 prompt。
            with gr.Row():
                # 创建提交按钮。
                msg = gr.Textbox(label="问题", lines=3, placeholder="点击发送", scale=20,
                                 show_label=False)
                db_wo_his_btn = gr.Button("发送", scale=1, icon="images/send.webp")

            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console", scale=1)
                chatbot.value = [[None, first]]

        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer_demo, inputs=[
            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
# 直接启动
demo.launch(share=True)
