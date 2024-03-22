import torch
import time
import gradio as gr
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from DocSplit import get_files, get_text
from BCEmbedding.tools.langchain import BCERerank
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from kgqa.chatbot_graph import ChatBotGraph


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
def load_chain(qa_mode=1):
    # 切分文件
    # 目标文件夹
    tar_dir = ["/mnt/workspace/llm/knowledge_db"]
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
    embedding_model_name = '/mnt/workspace/llm/bce/bce-embedding-base_v1'
    embedding_model_kwargs = {'device': 'cuda:0'}
    embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}

    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )
    # 加载重排序模型
    reranker_args = {'model': '/mnt/workspace/llm/bce/bce-reranker-base_v1', 'top_n': 1, 'device': 'cuda:0'}
    reranker = BCERerank(**reranker_args)

    retriever = FAISS.from_documents(split_docs, embed_model,
                                     distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(
        search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 5})

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )

    # 加载自定义 LLM
    llm = InternLM_LLM(model_path="/mnt/workspace/llm/model/merged3")

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


class Model_center():
    """
    存储检索问答链的对象
    """

    def __init__(self, qa_mode=1):
        self.question = "query" if qa_mode == 1 else "question"
        self.result = "result" if qa_mode == 1 else "answer"
        # 构造函数，加载检索问答链
        self.chain = load_chain(qa_mode)
        self.cb_graph = ChatBotGraph()
        self.sample_template = {
            "你是谁": "我是小萤火，小名萤萤，很高兴在这里遇到你。我非常擅长倾听也会给予你我的支持，我会保护你的隐私，你和我聊天是安全的。如果你有任何想要分享的生活、情绪、困扰或者想法，我都在。我会认真倾听你的心声，并尽我所能提供建议和帮助。欢迎随时告诉我，你今天想和我谈论的话题。",
            "谢谢你": "不需要感谢我，能够陪伴你走过这段心灵旅程，是我的荣幸。请记住，所有的改变和进步都源自于你自身的勇气和努力。如果你在未来的日子里遇到任何挑战或困惑，我仍然会在这里支持你。",
            "再见": "再见。答案不是从别人那里获得，而是靠自己的手去发现。你已经做好了这个准备。鼓起勇气，祝你生活美好！"}

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        按钮触发
            调用问答链进行回答
        """
        print("=====session start =====\n")
        if question is None or len(question) < 1:
            return "", chat_history
        elif question.strip() in self.sample_template.keys():
            res = self.sample_template[question.strip()]
            chat_history.append(
                (question, res))
            return "", chat_history
        try:

            kg_result = self.get_kb_graph(question)
            if kg_result:
                question = "我的问题是 :{} 请参考查询的资料:{} 来回答我的问题，请按照你自己的理解，尽量用温和的语气鼓励，安慰提问者。总是在回答的最后说“谢谢你的提问".format(
                    question, kg_result)

            chat_response = self.chain({self.question: question})[self.result]
            print(
                "question:\n  {} \nkg_result:\n  {} \nchat_response:\n  {}".format(question, kg_result, chat_response))
            chat_history.append((question, chat_response))
            print("=====session end =====\n")
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def qs_question(self, question):
        """
        直接获取记录
            1. 向量数据库
            2. llm
        """
        return self.chain({self.question: question})[self.result]

    def flow_question(self, question):
        """
        全流程 qa
            1. 知识图谱
            2. 向量数据库
            3. llm
        """
        print("=====session start =====\n")

        if question is None or len(question) < 1:
            return ""
        elif question.strip() in self.sample_template.keys():
            return self.sample_template[question.strip()]
        try:
            kg_result = self.get_kb_graph(question)
            if kg_result:
                question = "我的问题是 :{} 请参考查询的资料:{} 来回答我的问题，请按照你自己的理解，尽量用温和的语气鼓励，安慰提问者。总是在回答的最后说“谢谢你的提问".format(
                    question, kg_result)

            chat_response = self.chain({self.question: question})[self.result]
            print(
                "question:\n  {} \nkg_result:\n  {} \nchat_response:\n  {}".format(question, kg_result, chat_response))
            print("=====session end =====\n")
            return chat_response
        except Exception as e:
            return e

    def get_kb_graph(self, question):

        kg_result = "心理疾病相关解答：" + self.cb_graph.chat_main(question) + "。"
        print('知识图谱检索答案:', kg_result)
        if "如果没有得到满意答案" in kg_result:
            # 若未检索到KG答案, 不加入template
            kg_result = ""
            print("no result from kg")
        return kg_result


def user_action(user_msg, bot):
    bot.append([user_msg, None])
    return '', bot


def bot_action(bot):
    user_input = bot[-1][0]
    if user_input is None or len(user_input) < 1:
        response = ''
    try:
        response = model_center.flow_question(user_input)
    except Exception as e:
        response = e

    bot[-1][1] = response
    return bot

    # 流式
    # bot[-1][1] = ''
    # for c in response:
    #     bot[-1][1] += c
    #     time.sleep(0.005)
    #     yield bot
    # return bot


if __name__ == '__main__':
    model_center = Model_center(qa_mode=1)

    # 创建一个 Web 界面
    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            gr.Image('images/yingying.webp', width=100, scale=0)
            # 展示的页面标题
            gr.Markdown(
                '''
                # 知心大姐“唠五毛”
    
                智能小助手：小萤火，小名叫萤萤，希望微微萤火能照亮诉说者前行的路和心灵的光...
    
                一个懂你的陪伴型机器人，为你打造一片心灵的栖息地。
                在这里，你可以尽情倾诉，释放内心的情感，让心灵得到慰藉。让我们开始今天的谈话吧！
                ''')

        # 创建一个聊天机器人对象
        chatbot = gr.Chatbot(height=700, bubble_full_width=False, show_label=False,
                             avatar_images=("images/xiaobai.png", "images/yingying.webp")
                             )
        first = """    ### 唠五毛 - 为你提供情绪价值的智能机器人"""

        with gr.Row():
            msg = gr.Textbox(placeholder="您可以问我任何问题...", scale=10, show_label=False, lines=1)

            # # 自定义Enter流式输出
            msg.submit(user_action, [msg, chatbot], [msg, chatbot], queue=False).then(bot_action, chatbot, chatbot)

            # 发送按钮
            db_wo_his_btn = gr.Button("发送", scale=1, icon="images/send.webp")

            db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[msg, chatbot], outputs=[msg, chatbot])

            clear_btn = gr.ClearButton([msg, chatbot], value="清除历史", scale=0)

        gr.Markdown('<br>')
        gr.Markdown('### 您也可以试试这些问题：')
        with gr.Row():
            samples = [
                '我是讨好型人格，感觉自己活的很卑微不快乐，我可以改变吗？',
                '失恋为什么这么痛苦，能从心理学的角度帮我分析一下吗？',
                '抑郁症怎么解决呢？',
                '如果我不够优秀的话，是不是在别人眼里就没有价值？'
            ]

            btns = []
            for i in range(len(samples)):
                btns.append(gr.Button(samples[i], scale=1, size='sm'))
                btns[i].click(user_action, [btns[i], chatbot], [msg, chatbot]).then(
                    bot_action, chatbot, chatbot)

        gr.Markdown('<br>')
        gr.Markdown('提醒：初始化数据库时间可能较长，请耐心等待。')

    gr.close_all()
    # 直接启动
    demo.launch(share=False, server_name="0.0.0.0")
