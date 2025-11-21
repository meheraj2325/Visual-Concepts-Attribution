class PromptTemplates:
    """Centralized prompt templates for different tasks"""
    
    @staticmethod
    def answer_with_explanation(question: str) -> str:
        return f"""Question: {question}

                Please answer this question and provide a detailed explanation of your reasoning. 
                Explain what visual elements or information led you to this answer.

                Answer:"""
    
    @staticmethod
    def extract_visual_concepts() -> str:
        return """Analyze this image and list all the key visual concepts, objects, attributes, 
                and relationships you can identify. Be specific and comprehensive.

                Visual concepts:"""
    
    @staticmethod
    def extract_textual_keypoints(question: str) -> str:
        return f"""Text: {question}
                Extract the key ideas, important keywords, and noun phrases from this text. 
                List them clearly.

                Key points:"""