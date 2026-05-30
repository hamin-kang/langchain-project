from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
# 함수 기반 라우팅

load_dotenv()
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

def route_by_topic(input_dict):
    """주제에 따라 다른 체인 선택"""
    topic = input_dict.get("topic", "").lower()

    if "code" in topic or "programming" in topic:
        return "technical"
    elif "business" in topic or "strategy" in topic:
        return "business"
    else:
        return "general"
    
# 주제별 체인 정의
chains = {
    "technical": ChatPromptTemplate.from_template(
        "기술 전문가로서 답변: {question}"
    ) | llm | StrOutputParser(),

    "business": ChatPromptTemplate.from_template(
        "비즈니스 컨설턴트로서 답변: {question}"
    ) | llm | StrOutputParser(),
    "general": ChatPromptTemplate.from_template(
        "일반적인 관점에서 답변: {question}"
    ) | llm | StrOutputParser()
}

# 라우팅 체인
def routing_chain(input_dict):
    route = route_by_topic(input_dict)
    return chains[route].invoke(input_dict)

router = RunnableLambda(routing_chain)

# 테스트
result = router.invoke({
    "topic": "Python Programming",
    "question": "효율적인 파이썬 코드 작성 방법은?"
})

print(result)