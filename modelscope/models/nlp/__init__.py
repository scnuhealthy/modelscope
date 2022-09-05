# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbones import SbertModel
    from .heads import SequenceClassificationHead
    from .bert_for_sequence_classification import BertForSequenceClassification
    from .bert_for_document_segmentation import BertForDocumentSegmentation
    from .csanmt_for_translation import CsanmtForTranslation
    from .masked_language import (
        StructBertForMaskedLM,
        VecoForMaskedLM,
        BertForMaskedLM,
        DebertaV2ForMaskedLM,
    )
    from .nncrf_for_named_entity_recognition import (
        TransformerCRFForNamedEntityRecognition,
        LSTMCRFForNamedEntityRecognition)
    from .token_classification import SbertForTokenClassification
    from .sequence_classification import VecoForSequenceClassification, SbertForSequenceClassification
    from .space import SpaceForDialogIntent
    from .space import SpaceForDialogModeling
    from .space import SpaceForDialogStateTracking
    from .star_text_to_sql import StarForTextToSql
    from .task_models import (InformationExtractionModel,
                              SingleBackboneTaskModelBase)
    from .bart_for_text_error_correction import BartForTextErrorCorrection
    from .gpt3 import GPT3ForTextGeneration
    from .plug import PlugForTextGeneration
    from .sbert_for_faq_question_answering import SbertForFaqQuestionAnswering

else:
    _import_structure = {
        'star_text_to_sql': ['StarForTextToSql'],
        'backbones': ['SbertModel'],
        'heads': ['SequenceClassificationHead'],
        'csanmt_for_translation': ['CsanmtForTranslation'],
        'bert_for_sequence_classification': ['BertForSequenceClassification'],
        'bert_for_document_segmentation': ['BertForDocumentSegmentation'],
        'masked_language': [
            'StructBertForMaskedLM', 'VecoForMaskedLM', 'BertForMaskedLM',
            'DebertaV2ForMaskedLM'
        ],
        'nncrf_for_named_entity_recognition': [
            'TransformerCRFForNamedEntityRecognition',
            'LSTMCRFForNamedEntityRecognition'
        ],
        'palm_v2': ['PalmForTextGeneration'],
        'token_classification': ['SbertForTokenClassification'],
        'sequence_classification':
        ['VecoForSequenceClassification', 'SbertForSequenceClassification'],
        'space': [
            'SpaceForDialogIntent', 'SpaceForDialogModeling',
            'SpaceForDialogStateTracking'
        ],
        'task_models': [
            'InformationExtractionModel', 'SequenceClassificationModel',
            'SingleBackboneTaskModelBase'
        ],
        'bart_for_text_error_correction': ['BartForTextErrorCorrection'],
        'gpt3': ['GPT3ForTextGeneration'],
        'plug': ['PlugForTextGeneration'],
        'sbert_for_faq_question_answering': ['SbertForFaqQuestionAnswering'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
