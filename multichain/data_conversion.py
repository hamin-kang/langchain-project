from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 입력 변환 함수
def preprocess(input_dict):
    return {
        "processed_text": input_dict["text"].strip().lower(),
        "original": input_dict["text"]
    }

# 출력 변환 함수
def postprocess(output):
    return {
        "result": output,
        "length": len(output)
    }

chain = (
    RunnableLambda(preprocess)
    | ChatPromptTemplate.from_template("분석: {processed_text}")
    | llm
    | StrOutputParser()
    | RunnableLambda(postprocess)
)

result = chain.invoke({"text": "         앞으로 Intel과 AMD가 주도하는 CPU 시장에서 어떤 변화가 생길까?     "})

print(result)
