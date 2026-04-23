
from openai import OpenAI

def call_qwen(prompt, time=0):
    """
    使用千文大语言模型进行预测
    """
    try:
        client = OpenAI(
            api_key='sk-xx',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen3-max",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant for analyzing code.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("Error on attempt:", e)
        if time >= 3:
            return None  # 返回 None 以指示多次尝试后依然失败
        print("Retrying...")
        return call_qwen(prompt, time + 1)

