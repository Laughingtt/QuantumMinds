import gradio as gr
import random
import os
import json
import requests
import time


class WenXinChat:

    @staticmethod
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

    @staticmethod
    def wenxin_chat(user_input, history=[]):
        current_line = {'role': 'user', 'content': user_input}
        if len(history) == 0:
            prompt = [current_line]
            history = [current_line]
        else:
            prompt = history.append(current_line)

        response = WenXinChat.call_model(prompt)
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
            response, wenxin_history = WenXinChat.wenxin_chat(question, wenxin_history)
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
    openxlab.login('g4dz0p2pmw5vqvxxnpjz', '5d6m4l7vbp0ndra2pznql3y5egeyzorykxle1m8o')

    download(model_repo='DD-learning/llm', output='llm_model')
    print(os.listdir('.'))

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


# 下载模型
# download_model()


def user_action(user_msg, bot):
    bot.append([user_msg, None])
    return '', bot


def bot_action(bot):
    user_input = bot[-1][0]
    internlm_predict = load_InternLM_chain()
    if user_input == None or len(user_input) < 1:
        response = ''
    try:
        response = internlm_predict(user_input)
    except Exception as e:
        response = e

    bot[-1][1] = ''
    for c in response:
        bot[-1][1] += c
        time.sleep(0.005)
        yield bot
    return bot


SL="""
失恋之所以痛苦，可以从心理学的角度来分析。失恋时，人们会经历一系列强烈的情感反应，如失落、愤怒、悲伤、焦虑等。以下是一些心理学的解释：

1. 情感依恋：在一段恋爱关系中，人们往往会建立深厚的情感依恋。当关系结束时，这种依恋关系被打破，导致情感上的失落和痛苦。

2. 自我认知与自我价值：恋爱关系中的伴侣往往对彼此有积极的评价和认可。当失恋发生时，这种积极的自我认知和自我价值感会受到挑战，导致自尊心的受损和自我怀疑。

3. 认知偏差：失恋时，人们可能会陷入一些认知偏差，如过度解读分手的原因、过度自责等。这些认知偏差会加重痛苦和焦虑。

4. 恐惧与不安：失恋时，人们可能会感到恐惧和不安，担心未来是否能够找到新的伴侣或重新建立关系。这种不确定性和恐惧感也会加重痛苦。

5. 社会支持与归属感：恋爱关系中的伴侣往往在彼此的生活中扮演重要的角色，提供社会支持和归属感。当关系结束时，这种社会支持和归属感也会受到挑战，导致孤独和失落。

针对失恋带来的痛苦，可以尝试以下方法来缓解：

1. 寻求支持：与朋友、家人或专业人士交流，分享自己的感受和经历。

2. 自我关爱：关注自己的情感和身体需求，尝试一些放松的活动，如运动、冥想等。

3. 认知重构：尝试改变自己的认知模式，避免陷入认知偏差和消极思维。

4. 寻找新的兴趣和活动：尝试新的兴趣爱好或参与社交活动，以扩大社交圈和寻找新的归属感。

5. 寻求专业帮助：如果痛苦持续不减，可以寻求心理咨询或心理治疗等专业帮助。

最后，失恋虽然痛苦，但也是人生中的一部分经历。随着时间的推移，痛苦会逐渐减轻，你可以逐渐走出阴影，重新找到属于自己的幸福。
"""

def bot_action_demo(bot):
    user_input = bot[-1][0]

    response = random.choice([SL])

    bot[-1][1] = response
    return bot


# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        gr.Image('images/yingying.webp', width=100, scale=0,show_label=False, show_download_button=False)
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
        msg = gr.Textbox(placeholder="您可以问我任何问题...", scale=10, show_label=False)

        db_wo_his_btn = gr.Button("发送", scale=1, icon="images/send.webp")

        db_wo_his_btn.click(model_center.qa_chain_self_answer_demo, inputs=[
            msg, chatbot], outputs=[msg, chatbot])
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
                bot_action_demo, chatbot, chatbot)

    gr.Markdown('<br>')
    gr.Markdown('提醒：初始化数据库时间可能较长，请耐心等待。')

gr.close_all()
# 直接启动
demo.launch()
