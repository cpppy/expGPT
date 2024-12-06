from openai import OpenAI
import streamlit as st

def test_doubao():
    client = OpenAI(
        # api_key=os.environ.get("ARK_API_KEY"),
        api_key='cc39f7e2-359e-4acb-b2d5-09f219394316',
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    # # Non-streaming:
    # print("----- standard request -----")
    # completion = client.chat.completions.create(
    #     model="ep-20241031134418-mgbsw",  # your model endpoint ID
    #     messages=[
    #         {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
    #         {"role": "user", "content": "常见的十字花科植物有哪些？"},
    #     ],
    # )
    # print(completion.choices[0].message.content)

    # Streaming:
    print("----- streaming request -----")
    stream = client.chat.completions.create(
        model="ep-20241031134418-mgbsw",  # your model endpoint ID
        messages=[
            {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
            # {"role": "user", "content": "常见的十字花科植物有哪些？"},
            {"role": "user", "content": "今天上海的天气怎么样？"},
        ],
        stream=True
    )

def main():

    st.set_page_config(layout="wide")

    # st.title("gpt-3.5-turbo")

    # client = OpenAI()
    client = OpenAI(
        # api_key=os.environ.get("ARK_API_KEY"),
        api_key='cc39f7e2-359e-4acb-b2d5-09f219394316',
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    if "openai_model" not in st.session_state:
        # st.session_state["openai_model"] = "gpt-3.5-turbo"
        st.session_state["openai_model"] = "ep-20241031134418-mgbsw"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
