from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# 1. 컴포넌트 정의
topics = ["HBM", "HBF", "CXL"]
prompt = ChatPromptTemplate.from_template("반도체 시장에서 미래에 {topic}이 어떻게 될지 전망을 매우 간략히 알려줘.")
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
output_parser = StrOutputParser()

# 2. 체인 정의
chain = prompt | llm | output_parser

# 3. batch 실행
results = chain.batch([{"topic": t} for t in topics])
for topic, result in zip(topics, results):
    print(f"{topic}의 전망: {result}")