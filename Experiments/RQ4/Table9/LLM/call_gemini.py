from openai import OpenAI

client = OpenAI(
    api_key="sk-xx",
    base_url="https://xx/v1"
)


def call_gemini(prompt, time=0):
    messages = [
        {"role": "system", "content": "You are a helpful assistant for analyzing code."},
        {"role": "user", "content": prompt}
    ]

    model = "gemini-3.1-pro-preview"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content

    except Exception as e:
        print("Error on attempt:", e)
        if time >= 3:
            return None  # 返回 None 以指示多次尝试后依然失败
        print("Retrying...")
        return call_gemini(prompt, time + 1)
