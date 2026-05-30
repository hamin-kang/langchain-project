from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template("너는 천문학 전문가야. 질문에 답을 해줘. <Question>: {input}")

# LLM
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
output_parser = StrOutputParser()

# 체인 구성 (프롬프트 | LLM)
chain = prompt | llm | output_parser

# 체인 실행
response = chain.invoke({"input": "태양계에서 가장 큰 행성은 무엇인가?"})

print(response)