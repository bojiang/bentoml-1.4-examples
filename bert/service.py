import bentoml
from bentoml.models import HuggingFaceModel, BentoModel
from transformers import AutoModel
import time

@bentoml.service
class Iris:
    model = BentoModel("iris_clf:ggthayrc3wqnh4wa")
    tokenizer = HuggingFaceModel("bert-base-uncased")

    def __init__(self) -> None:
        start = time.time()
        print("bento model loaded", self.model)
        print("huggingface model loaded", self.tokenizer)
        self.model = AutoModel.from_pretrained(self.tokenizer)
        print("init time cost", time.time() - start)

    @bentoml.api
    def generate(self, prompt: str = "hello") -> str:
        return prompt + " world"

    @bentoml.on_shutdown
    def on_shutdown(self) -> None:
        print("on_shutdown")
