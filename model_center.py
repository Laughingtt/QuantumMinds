import copy

from kgqa.chatbot_graph import ChatBotGraph
from model.load_chain import load_chain


class Model_center(object):
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
                question_comb = "我的问题是 :{} 请参考查询的资料:{} 来回答我的问题，请按照你自己的理解，尽量用温和的语气鼓励，安慰提问者。总是在回答的最后说“谢谢你的提问".format(
                    question, kg_result)
            else:
                question_comb = copy.deepcopy(question)

            chat_response = self.chain({self.question: question_comb})[self.result]
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
                question_comb = "我的问题是 :{} 请参考查询的资料:{} 来回答我的问题，请按照你自己的理解，尽量用温和的语气鼓励，安慰提问者。总是在回答的最后说“谢谢你的提问".format(
                    question, kg_result)
            else:
                question_comb = copy.deepcopy(question)

            chat_response = self.chain({self.question: question_comb})[self.result]
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
