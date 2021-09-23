import tensorflow as tf
import torch


class LanguageTokens:
    """
    Static class to get the summarization task tokens
    """
    def __init__(self, tokenizer, tf_or_pt: str) -> None:
        """
        Init
        :param tokenizer: Tokenizer
        :type tokenizer: transformers Tokenizer
        :param tf_or_pt: "tf" or "pt" return tensorflow ot pytorch Tensor
        :type tf_or_pt: str
        """
        super().__init__()
        self.en_de_prefix = tokenizer("summarize English to German: ", return_tensors=tf_or_pt).input_ids
        self.de_en_prefix = tokenizer("summarize German to English: ", return_tensors=tf_or_pt).input_ids
        self.en_en_prefix = tokenizer("summarize English to English: ", return_tensors=tf_or_pt).input_ids
        self.de_de_prefix = tokenizer("summarize German to German: ", return_tensors=tf_or_pt).input_ids

        if tf_or_pt == "tf":
            self.en_de_prefix = tf.reshape(self.en_de_prefix, (-1,))
            self.de_en_prefix = tf.reshape(self.de_en_prefix, (-1,))
            self.en_en_prefix = tf.reshape(self.en_en_prefix, (-1,))
            self.de_de_prefix = tf.reshape(self.de_de_prefix, (-1,))
        elif tf_or_pt == "pt":
            self.en_de_prefix = self.en_de_prefix.reshape(-1,)
            self.de_en_prefix = self.de_en_prefix.reshape(-1,)
            self.en_en_prefix = self.en_en_prefix.reshape(-1,)
            self.de_de_prefix = self.de_de_prefix.reshape(-1,)

        # check if last token is end of sequence token and remove it
        if self.en_de_prefix[-1] == 1:
            self.en_de_prefix = self.en_de_prefix[:-1]
            self.de_en_prefix = self.de_en_prefix[:-1]
            self.en_en_prefix = self.en_en_prefix[:-1]
            self.de_de_prefix = self.de_de_prefix[:-1]

        assert self.en_de_prefix.shape[0] == self.de_en_prefix.shape[0] == self.en_en_prefix.shape[0] == self.de_de_prefix.shape[0], "All perfixes must have the same size"
        self.prefix_size = self.en_de_prefix.shape[0]