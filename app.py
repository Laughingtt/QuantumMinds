import gradio as gr
import random
import os
import json
import requests


from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA

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

def call_model(prompt):
    token = os.getenv('WToken')
    url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k'
    url += '?access_token=' + token

    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({
        "messages": prompt,
        "disable_search": False,
        "enable_citation": False
        # "max_output_tokens": 500
    })

    resp = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(resp.text)['result']

    return result


def wenxin_chat(user_input, history=[]):
    current_line = {'role': 'user', 'content': user_input}
    if len(history) == 0:
        prompt = [current_line]
        history = [current_line]
    else:
        prompt = history.append(current_line)

    response = call_model(prompt)
    history.append({'role': 'assistant', 'content': response})
    return response, history


def load_InternLM_chain():
    # 切分文件

    # 加载自定义 LLM
    llm = InternLM_LLM(model_path="model")

    # 定义一个 Prompt Template
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    return llm.predict

class Model_center():
    """
    存储检索问答链的对象
    """

    def __init__(self):
        # 构造函数，加载检索问答链
        # self.chain = load_chain()
        pass

    def qa_chain_self_answer_demo(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            bot_message = random.choice(["How are you?", "Hello Hello Hello", "I'm hungry"])
            bot_message = str(os.listdir(question))
            chat_history.append(
                (question, bot_message))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def qa_chain_self_answer_wenxin(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        wenxin_history = []
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            response, wenxin_history = wenxin_chat(question, wenxin_history)
            chat_history.append(
                (question, response))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history


    def qa_chain_self_answer_interlm(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        internlm_predict = load_InternLM_chain()
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            response = internlm_predict(question)
            chat_history.append(
                (question, response))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history

def download_model():
    from openxlab.model import download
    download(model_repo='DD-learning/llm', output='llm_model')
    download(model_repo='OpenLMLab/InternLM-chat-7b',output='model')
    print(os.listdir('.'))
    print(os.listdir('/home/xlab-app-center/llm_model'))
    print(os.listdir('model'))

def download_model2():
    base_path = './llm_model'
    # download repo to the base_path directory using git
    # os.system('apt install git')
    # os.system('apt install git-lfs')
    os.system(f'git clone https://code.openxlab.org.cn/DD-learning/model_demo.git {base_path}')
    os.system(f'cd {base_path} && git lfs pull')
    print(os.listdir('.'))
    os.system(f'cd ..')

download_model2()



# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>LLM</center></h1>
                <center>金科Demo</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="问题", lines=2, placeholder="Enter发送，Shift+Enter换行")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat", scale=1)
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer_interlm, inputs=[
            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
# 直接启动
demo.launch()
