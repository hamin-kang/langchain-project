from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# 여러 모델 초기화
gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
llama  = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

prompt = ChatPromptTemplate.from_template("{question}")

# 여러 모델 동시 호출
ensemble_chain = RunnableParallel(
    gemini=prompt | gemini | StrOutputParser(),
    llama=prompt | llama | StrOutputParser(),
)

# 결과 종합
synthesis_prompt = ChatPromptTemplate.from_template(
    """
    두 AI의 응답을 비교하고 최선의 답변을 종합하세요.

    Gemini의 답변:
    {gemini}

    Llama의 답변:
    {llama}

    종합 답변:
    """
)

final_chain = (
    ensemble_chain
    | synthesis_prompt
    | gemini
    | StrOutputParser()
)

result = final_chain.invoke({"question": "넥스트 엔비디아는 누가 될까요?"})

print(result)