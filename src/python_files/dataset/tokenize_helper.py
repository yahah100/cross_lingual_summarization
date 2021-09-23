from multiprocessing import Process, Manager
import tensorflow as tf


class TokenizeHelper:
    """
    Tokenize Helper to tokenize sequences for the T5 training.
    !!!!
    It is important to disable the GPU otherwise tensorflow will try to run this code on the GPU which results in errors
    !!!!
    """
    def __init__(self, tokenizer, prefix_size, num_threads=12) -> None:
        """
        Init
        :param tokenizer: Tokenizer
        :type tokenizer: transformer.tokenizer
        :param prefix_size: size of the task tokens
        :type prefix_size: int
        :param num_threads: Number of threads the tokenizer should run on
        :type num_threads: int
        """
        super().__init__()
        self.prefix_size = prefix_size
        self.tokenizer = tokenizer
        self.num_threads = num_threads

    def tokenize_articles(self, text):
        ids = self.tokenizer(text, max_length=(512 - self.prefix_size), truncation=True, padding='max_length',
                             return_tensors="tf")

        return tf.squeeze(ids.input_ids), tf.squeeze(ids.attention_mask)

    @staticmethod
    def shift_seq_right(seq):
        ones = tf.zeros([1], dtype=tf.int32)
        return tf.concat([ones, seq[:-1]], axis=0)

    @staticmethod
    def shift_seq_right_batch(seq):
        batch_size = seq.shape[0]
        ones = tf.zeros([batch_size, 1], dtype=tf.int32)
        return tf.concat([ones, seq[:, :-1]], axis=1)

    def tokenize_highlights(self, text):
        y = self.tokenizer(text, return_tensors="tf", max_length=150, truncation=True, padding='max_length').input_ids
        y = tf.squeeze(y)
        y_ids = self.shift_seq_right(y)

        return y, y_ids

    def get_tokenized_ds(self, articles, highlights):
        x = []
        x_mask = []
        for x_i in articles:
            t1, t2 = self.tokenize_articles(x_i)
            x.append(t1)
            x_mask.append(t2)

        y = []
        y_ids = []
        for y_i in highlights:
            t1, t2 = self.tokenize_highlights(y_i)
            y.append(t1)
            y_ids.append(t2)

        return x, x_mask, y, y_ids

    def split_ds(self, thread_index, articles, highlights):
        interval = int(len(articles) / self.num_threads)
        return articles[interval * thread_index: interval * (thread_index + 1)], highlights[
                                                                                 interval * thread_index: interval * (
                                                                                             thread_index + 1)]

    def p_tokenize(self, thread_index, articles, highlights, x, x_mask, y, y_ids):

        for i, x_i in enumerate(articles):
            t1, t2 = self.tokenize_articles(x_i)
            x[(thread_index, i)] = t1
            x_mask[(thread_index, i)] = t2

        for i, y_i in enumerate(highlights):
            t1, t2 = self.tokenize_highlights(y_i)
            y[(thread_index, i)] = t1
            y_ids[(thread_index, i)] = t2
        return x, x_mask, y, y_ids

    @staticmethod
    def dict_to_list(input_dict):
        return [item for key, item in sorted(input_dict.items())]

    def get_parallel_tokenized_ds(self, articles, highlights):
        """
        get parallel tokenized input text and output text of summarization
        :param articles: input text
        :type articles: list
        :param highlights: output text
        :type highlights: list
        :return: tuple of result lists of tokenized data
        :rtype: tuple
        """
        manager = Manager()

        x = manager.dict()
        x_mask = manager.dict()
        y = manager.dict()
        y_ids = manager.dict()

        coord = tf.train.Coordinator()
        processes = []
        for thread_index in range(self.num_threads):
            part_articles, part_highlights = self.split_ds(thread_index, articles, highlights)
            #             print("len:", len(part_articles), part_articles[0])
            args = (thread_index, part_articles, part_highlights, x, x_mask, y, y_ids)
            p = Process(target=self.p_tokenize, args=args)
            p.start()
            processes.append(p)
        coord.join(processes)
        return self.dict_to_list(x), self.dict_to_list(x_mask), self.dict_to_list(y), self.dict_to_list(y_ids)
