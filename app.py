import gradio as gr
import random
import os
import json
import requests



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
            # bot_message = str(os.listdir(question))
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
    import openxlab
    from openxlab.model import download
    openxlab.login('g4dz0p2pmw5vqvxxnpjz','5d6m4l7vbp0ndra2pznql3y5egeyzorykxle1m8o')

    download(model_repo='DD-learning/llm', output='llm_model')
    print(os.listdir('.'))

    # download(model_repo='OpenLMLab/InternLM-chat-7b',output='model')
    # print(os.listdir('.'))


    # from openxlab.dataset import get
    # get(dataset_repo='DD-learning/llm', target_path='llm_data')  # 数据集下载

    print(os.listdir('.'))
    print(os.listdir('/home/xlab-app-center/llm_model'))

def download_model2():
    base_path = './llm_model'
    # download repo to the base_path directory using git
    # os.system('apt install git')
    # os.system('apt install git-lfs')
    os.system(f'git clone https://code.openxlab.org.cn/DD-learning/model_demo.git {base_path}')
    os.system(f'cd {base_path} && git lfs pull')
    print(os.listdir('.'))
    os.system(f'cd ..')

# download_model()



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
demo.launch()