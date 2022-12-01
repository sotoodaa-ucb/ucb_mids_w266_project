from enum import Enum

from w266_project.models.baseline import MarkdownModel
from w266_project.models.code_markdown import CodeMarkdownModel, \
    CodeMarkdownModelV2


class ModelType(Enum):
    BASELINE = 'baseline'
    CODE_MARKDOWN = 'codemarkdown'
    CODE_MARKDOWN_V2 = 'codemarkdown_v2'


MODEL_TYPE_DICT = {
    # 768 -> 1
    ModelType.BASELINE: {
        'model': MarkdownModel,
        'checkpoint_path': 'models/artifacts/baseline_20221113_052917_0_0.1731',
        'url': 'https://drive.google.com/uc?id=1HDCzgmJCCzEpZdX1Cije8Zj4Mj7IiPtG'
    },
    # 1536 -> 1
    ModelType.CODE_MARKDOWN: {
        'model': CodeMarkdownModel,
        'checkpoint_path': 'models/artifacts/code-markdown-bert_20221119_115632_4_0.0605',
        'url': 'https://drive.google.com/uc?id=1u6zrVHF2ME2tatr9IEZDKwYNiVk6n2Km'
    },
    # 1536 -> 768 -> 1
    ModelType.CODE_MARKDOWN_V2: {
        'model': CodeMarkdownModelV2,
        'checkpoint_path': 'models/artifacts/code-markdown-bert_20221120_195204_0_0.17',
        'url': 'https://drive.google.com/uc?id=1hwAWibJbPkqBVaVgXOkBmcEzcdgU8M9g'
    }
}
