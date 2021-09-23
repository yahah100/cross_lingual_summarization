from os import listdir
import tensorflow as tf
# Taken from the TensorFlow models repository: https://github.com/tensorflow/models/blob/befbe0f9fe02d6bc1efb1c462689d069dae23af1/official/nlp/bert/input_pipeline.py#L24
class TFRecordLoader:
    """
    Load tfrecord files
    """

    def __init__(self, root_folder, language_tokens, prefix_size, MAX_ARTICLE_LEN, MAX_HIGHLIGHT_LEN, BATCH_SIZE) -> None:
        """
        Init
        :param root_folder: path to tfrecord files
        :type root_folder: str
        :param language_tokens: LanguageTokens
        :type language_tokens: LanguageTokens from language_tokens.py
        :param prefix_size: size of the summarization task for example "summarize English to English:"
        :type prefix_size: int
        :param MAX_ARTICLE_LEN: Max Article length
        :type MAX_ARTICLE_LEN: int
        :param MAX_HIGHLIGHT_LEN: Max Highlight length
        :type MAX_HIGHLIGHT_LEN: int
        :param BATCH_SIZE: Batchsize
        :type BATCH_SIZE: int
        """
        super().__init__()

        self.root_folder = root_folder
        self.language_tokens = language_tokens
        self.MAX_ARTICLE_LEN = MAX_ARTICLE_LEN
        self.MAX_HIGHLIGHT_LEN = MAX_HIGHLIGHT_LEN
        self.prefix_size = prefix_size
        self.BATCH_SIZE = BATCH_SIZE


    def get_tf_record_files(self, directory):
        """
        Read all record files in the directory
        :param directory: directory of tfrecord files
        :type directory: str
        :return: file list
        :rtype: list
        """
        file_list = []

        list_dir = listdir(directory)
        list_dir = [directory + "/" + item for item in list_dir]

        for item in list_dir:
            if item.split(".")[-1] == "tfrecord":
                file_list.append(str(item))
        return file_list

    def decode_record(self, record, features):
        """
        Decodes a record to a TensorFlow example
        :param record: Record
        :type record:
        :param features: dict with io types of tensorflow
        :type features: dict
        :return:
        :rtype:
        """
        example = tf.io.parse_single_example(record, features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        return example

    def select_data_from_record(self, record):
        """
        Combine record with prefix
        :param record: one record
        :type record: dict
        :return: tfrecord data
        :rtype: list
        """
        return [
            tf.concat([self.language_tokens.de_de_prefix, record['ger_x']], axis=0),
            tf.concat([tf.ones(self.prefix_size, dtype=tf.int32), record['ger_x_mask']], axis=0), record['ger_y'],
            record['ger_y_ids'],
            tf.concat([self.language_tokens.en_de_prefix, record['en_x']], axis=0),
            tf.concat([tf.ones(self.prefix_size, dtype=tf.int32), record['en_x_mask']], axis=0), record['ger_y'],
            record['ger_y_ids'],
            tf.concat([self.language_tokens.de_en_prefix, record['ger_x']], axis=0),
            tf.concat([tf.ones(self.prefix_size, dtype=tf.int32), record['ger_x_mask']], axis=0), record['en_y'],
            record['en_y_ids'],
            tf.concat([self.language_tokens.en_en_prefix, record['en_x']], axis=0),
            tf.concat([tf.ones(self.prefix_size, dtype=tf.int32), record['en_x_mask']], axis=0), record['en_y'],
            record['en_y_ids'],
        ]

    def get_tfrecord_dataset(self, folder):
        """
        Get the tf record dataset
        :param folder: path to tf record files
        :type folder: str
        :return: TFRecordDataset
        :rtype: TFRecordDataset
        """
        features = {
            'ger_x': tf.io.FixedLenFeature([self.MAX_ARTICLE_LEN - self.prefix_size], tf.int64),
            'ger_x_mask': tf.io.FixedLenFeature([self.MAX_ARTICLE_LEN - self.prefix_size], tf.int64),
            'ger_y': tf.io.FixedLenFeature([self.MAX_HIGHLIGHT_LEN], tf.int64),
            'ger_y_ids': tf.io.FixedLenFeature([self.MAX_HIGHLIGHT_LEN], tf.int64),

            'en_x': tf.io.FixedLenFeature([self.MAX_ARTICLE_LEN - self.prefix_size], tf.int64),
            'en_x_mask': tf.io.FixedLenFeature([self.MAX_ARTICLE_LEN - self.prefix_size], tf.int64),
            'en_y': tf.io.FixedLenFeature([self.MAX_HIGHLIGHT_LEN], tf.int64),
            'en_y_ids': tf.io.FixedLenFeature([self.MAX_HIGHLIGHT_LEN], tf.int64),
        }

        dataset = tf.data.TFRecordDataset(self.get_tf_record_files(self.root_folder + folder))

        dataset = dataset.map(lambda record: self.decode_record(record, features))
        dataset = dataset.map(self.select_data_from_record)
        return dataset.batch(self.BATCH_SIZE)