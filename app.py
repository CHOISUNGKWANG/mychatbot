import os
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="Azure OpenAI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 나만의 Azure OpenAI 챗봇")
st.caption("스트림릿으로 구현한 대화형 AI 도우미입니다.")

# 2. 사이드바에 하이퍼파라미터 설정 UI 구성
st.sidebar.header("🎛️ 모델 파라미터 설정")

pMsgCnt = st.sidebar.number_input("유지할 대화 세트 수 (pMsgCnt)", min_value=1, max_value=10, value=3, step=1)
pMaxTokens = st.sidebar.slider("Max Tokens (pMaxTokens)", min_value=100, max_value=16384, value=6553, step=100)
pTemperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
pTopP = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
pFrequencyPenalty = st.sidebar.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
pPresencePenalty = st.sidebar.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

# Stop 시퀀스 처리
stop_input = st.sidebar.text_input("Stop Sequences (쉼표로 구분, 예: \\n,5.)", value="")
pStop = [s.strip().replace("\\n", "\n") for s in stop_input.split(",") if s.strip()] if stop_input else None

# 대화 리셋 버튼
if st.sidebar.button("🧹 대화 기록 초기화"):
    st.session_state.messages = [
        {"role": "system", "content": "사용자가 정보를 찾는 데 도움이 되는 AI 도우미입니다."}
    ]
    st.rerun()

# 3. Azure OpenAI 클라이언트 초기화
load_dotenv(override=True)
endpoint = os.getenv("AZURE_OAI_ENDPOINT")
deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OAI_KEY")

@st.cache_resource
def get_openai_client():
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2025-01-01-preview",
    )

client = get_openai_client()

# 4. 세션 상태 초기화 (메시지 구조를 호환성이 높은 기본 문자열로 변경)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "사용자가 정보를 찾는 데 도움이 되는 AI 도우미입니다."}
    ]

# 5. 기존 대화 내용 화면에 출력
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# 6. 사용자 입력 창 및 챗봇 추론
if question := st.chat_input("질문을 입력하세요..."):

    # 사용자가 입력한 메시지 화면에 표시 및 세션에 저장
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({"role": "user", "content": question})

    # 대화 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                completion = client.chat.completions.create(
                    model=deployment,
                    messages=st.session_state.messages,
                    max_tokens=pMaxTokens,
                    temperature=pTemperature,
                    top_p=pTopP,
                    frequency_penalty=pFrequencyPenalty,
                    presence_penalty=pPresencePenalty,
                    stop=pStop,
                    stream=False
                )

                answer = completion.choices[0].message.content
                st.write(answer)

                # 어시스턴트 답변 세션에 저장
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # 7. [위치 변경] 대화 세트 제한 규칙 적용 후 즉시 화면 새로고침
                # 답변이 추가된 직후 검사하여 오래된 대화를 잘라냅니다.
                while len(st.session_state.messages) > 1 + pMsgCnt * 2:
                    st.session_state.messages.pop(1)
                
                # 데이터가 완전히 정리된 상태로 화면을 다시 그립니다.
                st.rerun()

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
