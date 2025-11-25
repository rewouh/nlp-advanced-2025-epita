from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class NPCBrain:
    def __init__(self, char_name, context, model="qwen2.5:3b"):
        # On utilise qwen2.5:3b par défaut (léger pour tout le monde)
        print(f"Loading Brain with model: {model}...")
        self.llm = ChatOllama(model=model, temperature=0.8) # 0.8 pour un peu plus de créativité en RP
        
        # Mémoire tampon (garde l'historique brut)
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Prompt optimisé pour les petits modèles (Qwen/Llama3)
        # On lui dit d'être concis, sinon les petits modèles ont tendance à radoter.
        template = """You are {char_name}. {context}
        Current situation: A player is talking to you.
        Instructions: 
        - Answer directly in character.
        - Keep it short (1 or 2 sentences).
        - No actions between *asterisks*, just dialogue.
        
        Conversation history:
        {chat_history}
        
        Player: {human_input}
        {char_name}:"""
        
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            partial_variables={"char_name": char_name, "context": context},
            template=template
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def think(self, text):
        return self.chain.run(human_input=text)