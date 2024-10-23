import torch
from langchain.tools import BaseTool
from transformers import RobertaTokenizer
from models.linear_classifier import LinearConfig, LinearClassifier
from typing import Optional
from pydantic import PrivateAttr


class LinearClassifierTool(BaseTool):

    name: str = "RoBIn Classifier"
    description: str = """A linear classifier for risk of bias inference task. ALWAYS use this tool 
        when asked to evaluate or assess the risk of bias. Useful for risk of bias evaluation/assessment/inference.
        Use the entire prompt as input to the tool. For instance, if the prompt is
        "Evaluate the risk of bias in this study. The study is about the effects of a new drug on patients with
        diabetes.", the input should be "Evaluate the risk of bias in this study. The study is about the effects of a new drug on patients with
        diabetes.".
        """
    
    _tokenizer: RobertaTokenizer = PrivateAttr()
    _model: torch.nn.Module = PrivateAttr()
    
    def __init__(self, model_name: Optional[str] = "allenai/biomed_roberta_base"):
        super().__init__()
        self._tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self._model = LinearClassifier.from_pretrained("/models/linear_classifier_4e-05/pretrained")
        # self._model = LinearClassifier(LinearConfig(model_name=model_name,
        #                                             num_classes=2,
        #                                             dropout=0.2,
        #                                             pos_weight=0.5,
        #                                             loss_fn_cls='ce'))

    def _run(self, text: str) -> str:

        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self._model(**inputs)

        predicted_class = torch.argmax(outputs[1], dim=1).item()

        return "High/Unclear risk of bias" if predicted_class == 0 else 'Low risk of bias'
    

    async def _arun(self, text: str) -> str:
        raise NotImplementedError("This tool does not support async execution.")


