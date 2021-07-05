# coding=utf-8

"""
SensiMix: Sensitivity-Aware 8-bit Index & 1-bit Value Mixed Precision Quantization for BERT Compression
Authors:
- Tairen Piao (piaotairen@snu.ac.kr), Seoul National University
- Ikhyun Cho (ikhyuncho@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

""" Auto Config class. """
import logging
from collections import OrderedDict

from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_mpqbert import MPQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MPQBertConfig
from transformers.configuration_utils import PretrainedConfig



logger = logging.getLogger(__name__)


ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = dict(
    (key, value)
    for pretrained_map in [
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ]
    for key, value, in pretrained_map.items()
)


CONFIG_MAPPING = OrderedDict(
    [
        ("bert", BertConfig,),
        ("mpqbert",MPQBertConfig,),# piao
    ]
)


class AutoConfig:
    r"""
        :class:`~transformers.AutoConfig` is a generic configuration class
        that will be instantiated as one of the configuration classes of the library
        when created with the :func:`~transformers.AutoConfig.from_pretrained` class method.

        The :func:`~transformers.AutoConfig.from_pretrained` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type, *args, **kwargs):
        for pattern, config_class in CONFIG_MAPPING.items():
            if pattern in model_type:
                return config_class(*args, **kwargs)
        raise ValueError(
            "Unrecognized model identifier in {}. Should contain one of {}".format(
                model_type, ", ".join(CONFIG_MAPPING.keys())
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiates one of the configuration classes of the library
        from a pre-trained model configuration.

        The configuration class to instantiate is selected
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string.
            - contains `distilbert`: :class:`~transformers.DistilBertConfig` (DistilBERT model)
            - contains `albert`: :class:`~transformers.AlbertConfig` (ALBERT model)
            - contains `xlm-roberta`: :class:`~transformers.XLMRobertaConfig` (XLM-RoBERTa model)
            - contains `roberta`: :class:`~transformers.RobertaConfig` (RoBERTa model)
            - contains `bert`: :class:`~transformers.BertConfig` (Bert model)
            - contains `openai-gpt`: :class:`~transformers.OpenAIGPTConfig` (OpenAI GPT model)
            - contains `gpt2`: :class:`~transformers.GPT2Config` (OpenAI GPT-2 model)


        Args:
            pretrained_model_name_or_path (:obj:`string`):
                Is either: \
                    - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g.: ``bert-base-uncased``.
                    - a string with the `identifier name` of a pre-trained model configuration that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                    - a path to a `directory` containing a configuration file saved using the :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                    - a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.

            cache_dir (:obj:`string`, optional, defaults to `None`):
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download (:obj:`boolean`, optional, defaults to `False`):
                Force to (re-)download the model weights and configuration files and override the cached versions if they exist.

            resume_download (:obj:`boolean`, optional, defaults to `False`):
                Do not delete incompletely received file. Attempt to resume the download if such a file exists.

            proxies (:obj:`Dict[str, str]`, optional, defaults to `None`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`.
                The proxies are used on each request. See `the requests documentation <https://requests.readthedocs.io/en/master/user/advanced/#proxies>`__ for usage.

            return_unused_kwargs (:obj:`boolean`, optional, defaults to `False`):
                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part of kwargs which has not been used to update `config` and is otherwise ignored.

            kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`): key/value pairs with which to update the configuration object after loading.
                - The values in kwargs of any keys which are configuration attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.


        Examples::

            config = AutoConfig.from_pretrained('bert-base-uncased')  # Download configuration from S3 and cache.
            config = AutoConfig.from_pretrained('./test/bert_saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = AutoConfig.from_pretrained('./test/bert_saved_model/my_configuration.json')
            config = AutoConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = AutoConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, _ = PretrainedConfig.get_config_dict(
            pretrained_model_name_or_path, pretrained_config_archive_map=ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, **kwargs
        )

        if "model_type" in config_dict:
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            # Fallback: use pattern matching on the string.
            for pattern, config_class in CONFIG_MAPPING.items():
                if pattern in pretrained_model_name_or_path:
                    return config_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            "Unrecognized model in {}. "
            "Should have a `model_type` key in its config.json, or contain one of the following strings "
            "in its name: {}".format(pretrained_model_name_or_path, ", ".join(CONFIG_MAPPING.keys()))
        )
