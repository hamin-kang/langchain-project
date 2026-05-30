from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 가상의 검색 함수
def retrieve_context(query: str) -> str:
    # 실제로는 벡터 스토어 검색
    return f"검색된 컨텍스트: {query}에 대한 정보 ..."

# RAG 체인 구성
rag_chain = (
    # 1. 쿼리와 컨텍스트 병렬 준비
    RunnableParallel(
        question=RunnablePassthrough(),
        context=RunnableLambda(lambda x: retrieve_context(x["question"]))
    )
    # 2. 프롬프트 생성
    | ChatPromptTemplate.from_template(
        """
        컨텍스트를 참고하여 질문에 답변하세요.

        컨텍스트: {context}

        질문: {question}

        답변: 
        """
    )
    # 3. LLM 호출
    | llm
    # 4. 출력 파싱
    | StrOutputParser()
)

result = rag_chain.invoke({"question": "LangChain이란?"})
print(result)