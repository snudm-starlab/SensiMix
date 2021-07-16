"""
SensiMix: Sensitivity-Aware 8-bit Index & 1-bit Value Mixed Precision Quantization for BERT Compression
Authors:
- Tairen Piao (piaotairen@snu.ac.kr), Seoul National University
- Ikhyun Cho (ikhyuncho@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

__version__ = "2.5.1"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, AutoConfig
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_mpqbert import MPQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MPQBertConfig

# Configurations
from transformers.configuration_utils import PretrainedConfig
from transformers import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    is_sklearn_available,
)

# Files and general utilities
from transformers import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_tf_available,
    is_torch_available,
)

# Model Cards
from transformers import ModelCard

# Pipelines
from transformers import (
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    QuestionAnsweringPipeline,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    pipeline,
)
from transformers import AutoTokenizer
from transformers import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer



logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


if is_sklearn_available():
    from transformers import glue_compute_metrics, xnli_compute_metrics


# Modeling
if is_torch_available():
    from transformers.modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering
    from transformers import (
        AutoModel,
        AutoModelForPreTraining,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelWithLMHead,
        AutoModelForTokenClassification,
        ALL_PRETRAINED_MODEL_ARCHIVE_MAP,
    )

    # Import the BERT model
    from transformers import (
        BertPreTrainedModel,
        BertModel,
        BertForPreTraining,
        BertForMaskedLM,
        BertForNextSentencePrediction,
        BertForSequenceClassification,
        BertForMultipleChoice,
        BertForTokenClassification,
        BertForQuestionAnswering,
        load_tf_weights_in_bert,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    )

    # Import the SensiMix model
    from .modeling_mpqbert import (
        MPQBertPreTrainedModel,
        MPQBertModel,
        MPQBertForPreTraining,
        MPQBertForMaskedLM,
        MPQBertForSequenceClassification,
        MPQBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    )

    # Import the SensiMix model for inference
    from .modeling_mpqbert_infer import (
        MPQBertPreTrainedModel,
        MPQBertModel,
        MPQBertForPreTraining,
        MPQBertForMaskedLM,
        MPQBertForSequenceClassification_inference,
        MPQBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    )

    # Optimization
    from .optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )



if not is_tf_available() and not is_torch_available():
    logger.warning(
        "Neither PyTorch nor TensorFlow >= 2.0 have been found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
