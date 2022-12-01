import gdown
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch

from w266_project.models.core import MODEL_TYPE_DICT, ModelType
from w266_project.preprocessor.core import Preprocessor


class Engine(ABC):
    """Abstract base class of an inference engine. All possible inference
    engines including PyTorch, ONNX, TensorRT, etc, should extend this
    object. Should contain all common logic related to preprocessing and
    inference for the various inference engines.
    Attributes:
        model_type (ModelType): The type of model (enum).
    """

    def __init__(self, model_type: ModelType):
        if model_type not in MODEL_TYPE_DICT:
            raise ValueError(f'Invalid model type provided: {model_type}')

        self.model_type = model_type
        self.model = MODEL_TYPE_DICT[model_type]['model']()
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), MODEL_TYPE_DICT[model_type]['checkpoint_path'])
        self.model_url = MODEL_TYPE_DICT[model_type]['url']

        if not os.path.exists(self.checkpoint_path):
            print(f'{self.checkpoint_path} does not exist... downloading.')
            gdown.download(self.model_url, output=self.checkpoint_path)

    def preprocess(self, markdown_inputs: Union[List[str], str], code_inputs: List[str]):

        if self.model_type == ModelType.BASELINE:
            preprocessor = Preprocessor(
                markdown_tokenizer=self.model.code_tokenizer,
                code_tokenizer=self.model.code_tokenizer
            )

        return preprocessor.preprocess(
            markdown_inputs=markdown_inputs,
            code_inputs=code_inputs,
        )

    @abstractmethod
    def predict(self, image) -> Tuple[int, str]:
        raise NotImplementedError()


class PyTorchEngine(Engine):
    def __init__(self, model_type: ModelType):
        super(PyTorchEngine, self).__init__(model_type)

        # Only PyTorch runtime requires a device specification.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Instantiate the proper model and obtain checkpoint.
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=torch.device(self.device)
        )

        # Load from checkpoint.
        self.model.load_state_dict(state_dict=self.checkpoint)

        # Set model to evaluation mode.
        self.model.eval()

    def predict(
        self,
        ids_tensor: List[int],
        mask_tensor: List[List[int]],
        feature_tensor: torch.Tensor = None
    ):
        # Evaluation mode improves inference performance by turning off BatchNorms and Dropouts.
        self.model.eval()

        # Inference, unsqueeze to convert to 2D tensor from 1D tensor.
        if feature_tensor:
            raw_prediction = self.model(ids_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), feature_tensor.unsqueeze(0))
        else:
            raw_prediction = self.model(ids_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))

        return raw_prediction
