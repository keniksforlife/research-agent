import base64
import openai
import os


def main():

    image_analysis_prompt = """
      Please analyze the image carefully. Consider the following aspects to determine whether it is more likely a lifestyle image or a product image:

      1. Context: Is the product shown in a real-life setting or scenario? Are there people interacting with it in a way that suggests everyday use or a certain lifestyle?
      2. Focus: Is the primary focus on the product itself, with clear, detailed views of its features, or is the product part of a larger scene or narrative?
      3. Emotional Appeal: Does the image seem to be telling a story or creating an emotional connection, suggesting how the product enhances a particular lifestyle or experience?
      4. Background: Is the product displayed against a plain and neutral background or within a setting that adds context to its use?

      Based on these criteria, is this image a lifestyle image (Yes/No)? Please only answer with Yes or No.
      """

    # Craft the prompt for GPT
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": image_analysis_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://m.media-amazon.com/images/I/71g1fX6P0FL._AC_SL1500_.jpg"
                    }
                }
            ]
        }
    ]

    # Send a request to GPT
    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "api_key": "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",
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 1000,
    }

    result = openai.ChatCompletion.create(**params)
    response = result.choices[0].message.content.strip().lower()

    # Check if the response contains 'yes' or 'no' and return accordingly
    if 'yes' in response:
        return False
    elif 'no' in response:
        return True
    else:
        print("Response does not contain a clear Yes or No.")
        return None

if __name__ == "__main__":
    print(main())