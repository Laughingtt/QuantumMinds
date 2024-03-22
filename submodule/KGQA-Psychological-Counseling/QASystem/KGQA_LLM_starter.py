from langchain.vectorstores import Chroma
import os
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from DocSplit import get_files, get_text
from BCEmbedding.tools.langchain import BCERerank
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatbot_graph import ChatBotGraph


def load_chain():
    # 切分文件
    # 目标文件夹
    tar_dir = ["/mnt/workspace/pre_knowledge_db"]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # 加载问答链
    # init embedding model
    embedding_model_name = '/mnt/workspace/models/bce-embedding-base_v1'
    embedding_model_kwargs = {'device': 'cuda:0'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )
    # 加载重排序模型
    reranker_args = {'model': '/mnt/workspace/models/bce-reranker-base_v1', 'top_n': 1, 'device': 'cuda:0'}
    reranker = BCERerank(**reranker_args)
    # embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

    # example 1. retrieval with embedding and reranker
    retriever = FAISS.from_documents(split_docs, embed_model,
                                     distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
        search_type="similarity", search_kwargs={"score_threshold": 0.5, "k": 5})

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )

    # 加载自定义 LLM
    llm = InternLM_LLM(model_path="/mnt/workspace/models/merged3")
    # llm = InternLM_LLM(model_path="/mnt/workspace/models/merged")
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    # conversation_llm = ConversationChain(
    #     llm=llm,
    #     verbose=True,
    #     memory=ConversationBufferMemory()
    # )

    def qa_chain(question, kg_result):
        # 定义一个 Prompt Template
        template = """请参考以下上下文来回答最后的问题。如果上下文的内容不相关，请按照你自己的理解，尽量用温和的语气鼓励，安慰提问者。总是在回答的最后说“谢谢你的提问！”。
        {context}。{kg_result}
        问题: {question}
        有用的回答:"""

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                         partial_variables={"kg_result": kg_result}, template=template)

        # 运行 chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever, return_source_documents=True,
                                               chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        return qa_chain({"query": question})

    return qa_chain


class Model_center():
    """
    存储检索问答链的对象
    """

    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()
        self.handler = ChatBotGraph()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """

        sample_template = {
            "你是谁": "我是小萤火，小名萤萤，很高兴在这里遇到你。我非常擅长倾听也会给予你我的支持，我会保护你的隐私，你和我聊天是安全的。如果你有任何想要分享的生活、情绪、困扰或者想法，我都在。我会认真倾听你的心声，并尽我所能提供建议和帮助。欢迎随时告诉我，你今天想和我谈论的话题。",
            "谢谢你": "不需要感谢我，能够陪伴你走过这段心灵旅程，是我的荣幸。请记住，所有的改变和进步都源自于你自身的勇气和努力。如果你在未来的日子里遇到任何挑战或困惑，我仍然会在这里支持你。",
            "再见": "再见。答案不是从别人那里获得，而是靠自己的手去发现。你已经做好了这个准备。鼓起勇气，祝你生活美好！"}

        kg_result = "心理疾病相关解答：" + self.handler.chat_main(question) + "。"
        print('知识图谱检索答案:', kg_result)
        if "如果没有得到满意答案" in kg_result:
            # 若未检索到KG答案, 不加入template
            kg_result = ""
            print("no result from kg")

        if question == None or len(question) < 1:
            return "", chat_history
        elif question.strip() in sample_template.keys():
            res = sample_template[question.strip()]
            chat_history.append(
                (question, res))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        try:
            # res = self.chain({"query": question})
            res = self.chain(question, kg_result)
            print(res)
            chat_history.append(
                (question, res["result"]))
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
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>书生浦语</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
# 直接启动
# demo.launch(share=True)
demo.launch(share=True)
