class Interviewer:
    def interview(self, questions):
        responses = []
        for question in questions:
            response = self.ask_question(question)
            responses.append(response)
        return responses

    def ask_question(self, question):
        # 实现提问逻辑
        # 包括追问和轮数限制
        pass
