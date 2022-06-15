"""
Parts based on https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
"""
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class BertWrapper:
    def __init__(self, bert_max_len,num_classes):
        self.bert_max_len = bert_max_len
        self.num_classes = num_classes
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", do_lower_case=True
        )

        self.model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=self.num_classes,
            output_attentions=False,
            output_hidden_states=False,
        )


    def pre_pro(self, text):
        assert isinstance(text, list)

        tokens = self.tokenizer.encode_plus(
            " ".join(text),
            max_length=self.bert_max_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
        )

        return tokens["input_ids"], tokens["attention_mask"]
