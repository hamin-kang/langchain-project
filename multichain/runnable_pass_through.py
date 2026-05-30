from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 입력 데이터 보존, 원본 입력을 유지하면서 추가 데이터 할당
chain = (
    RunnablePassthrough.assign(
        # summary 키에 요약 결과 추가
        summary=ChatPromptTemplate.from_template("{text}를 요약") | llm | StrOutputParser()
    )
    | RunnablePassthrough.assign(
        # keywords 키에 키워드 추가
        keywords=ChatPromptTemplate.from_template("{text}의 키워드") | llm | StrOutputParser()
    )
)

result = chain.invoke({
    "text": """
[이데일리 한광범 기자] 과학기술정보통신부는 류제명 제2차관을 수석대표로 5월 29일(현지 시간) 프랑스 파리에서 개최된 주요 7개국(G7) 디지털 기술 장관회의에 참석했다고 밝혔다. 한국은 지난해에 이어 올해도 디지털·인공지능(AI) 분야 주요 협력국으로 G7 회의에 초청받았다.

이번 회의에는 G7 회원국과 한국, 스위스, 인도, 브라질, 케냐 등 초청국, 국제기구 고위급 인사들이 참석해 안전하고 책임 있는 AI, 중소기업의 AI 활용 확산, 디지털 기반 탄소중립, 온라인 미성년자 보호 등 디지털 기술 분야의 주요 현안과 협력 방안을 논의했다.

류제명 제2차관은 초청국 세션에서 “디지털·AI 대전환의 시기에 필요한 것은 혁신과 신뢰 중 하나를 선택하는 것이 아니라, 두 가지를 함께 진전시키는 것”이라며 G7 디지털 기술 장관회의의 4대 우선 의제와 관련한 한국의 정책 방향을 소개했다.

중소기업의 AI 도입과 관련해서는 ‘AI 원스톱 바우처’, AX 전문가 양성 프로그램, AI 데이터센터 기반 지원 등 한국의 AI 확산 정책과 스타트업·중소기업의 혁신 및 사업화 지원 방향을 설명했다.

이어 올해 시행된 「인공지능 발전과 신뢰 기반 조성 등에 관한 기본법(인공지능 기본법)」과 AI 안전연구소, ‘AI 서울 정상회의’, ‘아시아태평양경제협력체(APEC) AI 이니셔티브’ 등을 소개하며 AI 거버넌스 방향을 공유했다.

이와 함께 데이터센터 저전력화, 저전력 AI 네트워크, 친환경 AI 데이터센터 등 디지털 기술 기반의 탄소중립 정책과 AI 기반 아동·청소년 온라인 성착취 선제 대응 시스템을 설명했다.

류 차관은 회의 기간 중 미국, 일본, 영국 등 주요국 대표들과 만나 AI, 양자 등 과학기술 및 디지털 분야의 협력 방안을 논의했다.

과기정통부는 이번 회의 참석을 계기로 프랑스 AI 기업 미스트랄(Mistral) AI를 방문해 아서 멘슈 최고경영자(CEO)와 면담을 가질 예정이다. 이번 면담은 지난 4월 프랑스 대통령 국빈 방문 당시 방한한 미스트랄 AI 최고경영자와의 후속 논의로, 양국 간 AI 산업 협력, 인재 교류, 인공지능 기본법 발전 방향 협력 등을 논의할 계획이다.

류 차관은 “AI와 디지털 기술은 새로운 성장의 기회를 창출하는 동시에 신뢰와 안전이라는 과제도 함께 제기하고 있다”며 “이번 G7 디지털 기술 장관회의 참석을 계기로 혁신과 신뢰의 균형을 바탕으로 글로벌 AI 협력에 적극 기여해 나가겠다”고 밝혔다.
    """})

print(result)