import bentoml
from bentoml.models import HuggingFaceModel
import time


@bentoml.service
class Iris:
    tokenizer = HuggingFaceModel("bert-base-uncased")

    def __init__(self) -> None:
        from transformers import AutoModel

        start = time.time()
        print("huggingface model loaded", self.tokenizer)
        self.model = AutoModel.from_pretrained(self.tokenizer)
        print("init time cost", time.time() - start)

    @bentoml.api
    def generate(self, prompt: str = "hello") -> str:
        return prompt + " world"

    @bentoml.on_shutdown
    def on_shutdown(self) -> None:
        print("on_shutdown")
