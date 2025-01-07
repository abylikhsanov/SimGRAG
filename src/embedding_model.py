from sentence_transformers import SentenceTransformer
from openai import OpenAI, AzureOpenAI, RateLimitError
import os
import logging
from itertools import islice
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter, stop_after_delay

class EmbeddingModel:
    def __init__(self, configs):
        self.model = os.getenv("AZURE_OPENAI_EMBEDDING_NAME")
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_delay(3600),
        before_sleep=lambda retry_state: logging.warning(
            f"Rate limit exceeded for Embedding. Retrying in {retry_state.next_action.sleep} seconds..."
        ),
    )
    def encode(self, texts, batch_size=64, show_progress_bar=False):
        def batch(iterable, size):
            iterator = iter(iterable)
            for first in iterator:
                yield [first] + list(islice(iterator, size - 1))

        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text_batch in batch(texts, batch_size):
            response = self.client.embeddings.create(
                input=text_batch,
                model=self.model,
            )
            embeddings.extend([embedding_item.embedding for embedding_item in response.data])

        # Return single embedding if input was a single string
        return embeddings[0] if len(embeddings) == 1 else embeddings