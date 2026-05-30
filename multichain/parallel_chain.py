from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
# 병렬 체인

load_dotenv()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 세 가지 관점에서 동시 분석
positive_prompt = ChatPromptTemplate.from_template(
    "{topic}의 긍정적인 측면 3가지를 설명하시오."
)

negative_prompt = ChatPromptTemplate.from_template(
    "{topic}의 부정적인 측면 3가지를 설명하시오."
)

neutral_prompt = ChatPromptTemplate.from_template(
    "{topic}에 대한 객관적인 현황을 설명하시오."
)

# 병렬 체인 구성
parallel_chain = RunnableParallel(
    positive=positive_prompt | llm | StrOutputParser(),
    negative=negative_prompt | llm | StrOutputParser(),
    neutral=neutral_prompt | llm | StrOutputParser()
)

# 실행 (세 체인이 동시에 실행됨)
result = parallel_chain.invoke({"topic": "원격 근무"})

print("============ 긍정적인 측면 =============")
print(result["positive"])
print("============ 부정적인 측면 =============")
print(result["negative"])
print("============ 객관적인 현황 =============")
print(result["neutral"])