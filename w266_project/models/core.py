from enum import Enum

from w266_project.models.baseline import MarkdownModel


class ModelType(Enum):
    BASELINE = 'baseline'


MODEL_TYPE_DICT = {
    ModelType.BASELINE: {
        'model': MarkdownModel,
        'checkpoint_path': 'models/artifacts/baseline_20221113_052917_0_0.1731',
        'url': 'https://drive.google.com/uc?id=1HDCzgmJCCzEpZdX1Cije8Zj4Mj7IiPtG'
    }
}
