from openai import OpenAI, AzureOpenAI
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter, stop_after_delay
import logging
import os

class LLM:
    def __init__(self, configs):
        self.configs = configs['llm']
        print('Loading LLM ...')
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        print(self.chat("Hello!"))

    @retry(
        retry=retry_if_exception_type(ResourceExhausted),
        wait=wait_exponential_jitter(initial=1, max=10),
        stop=stop_after_delay(3600),
        before_sleep=lambda retry_state: logging.warning(
            f"Rate limit exceeded for query_gemini. Retrying in {retry_state.next_action.sleep} seconds..."
        ),
    )
    def chat(self, input_text):
        #completion = self.client.chat.completions.create(
        #    model=self.configs['model'],
        #    messages=[{"role":"user", "content":input_text}],
        #    temperature=self.configs['temperature'],
        #    top_p=self.configs['top_p'],
        #    max_tokens=self.configs['max_tokens']
        #)
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
        response = model.generate_content(input_text)
        return response.text
        #return completion.choices[0].message.content.strip()