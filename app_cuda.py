import time
import gradio as gr
from model_center import Model_center


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


def bot_action_stream(bot):
    user_input = bot[-1][0]
    if user_input is None or len(user_input) < 1:
        response = ''
    try:
        response = model_center.flow_question(user_input)
    except Exception as e:
        response = e

    # 流式
    bot[-1][1] = ''
    for c in response:
        bot[-1][1] += c
        time.sleep(0.005)
        yield bot
    return bot


def run_gr():
    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            gr.Image('images/yingying.webp', width=100, scale=0, show_label=False, show_download_button=False)
            # 展示的页面标题
            gr.Markdown(
                '''
                # 知心大姐“唠五毛”

                智能小助手：小萤火，小名叫萤萤，希望微微萤火能照亮诉说者前行的路和心灵的光...

                一个懂你的陪伴型机器人，为你打造一片心灵的栖息地。
                在这里，你可以尽情倾诉，释放内心的情感，让心灵得到慰藉。让我们开始今天的谈话吧！
                ''')

        # 创建一个聊天机器人对象
        chatbot = gr.Chatbot(height=550, bubble_full_width=False, show_label=False,
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


if __name__ == '__main__':
    # 创建一个 Web 界面
    model_center = Model_center(qa_mode=2)

    run_gr()
