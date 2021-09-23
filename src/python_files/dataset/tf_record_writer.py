from pathlib import Path
import tensorflow as tf
import time

from multiprocessing import Process, Manager


class TfRecordWriter:
    """
    Class to write multithreaded tf record files
    """
    def __init__(self, num_threads) -> None:
        """
        Init
        :param num_threads: Number of threads
        :type num_threads: int
        """
        super().__init__()

        self.num_threads = num_threads

    def split_ds(self, i, ds):
        interval = int(len(ds) / self.num_threads)
        return ds.skip(interval * i).take(interval)

    @staticmethod
    def write_instances_to_tfrecord(thread_index, features_dataset, folder, file_name):
        with tf.io.TFRecordWriter(f"{folder}/{file_name}-{thread_index}.tfrecord") as tfwriter:
            start_time = time.time()
            log_interval = 1000
            for i, train_feature in enumerate(features_dataset):
                (ger_x, ger_x_mask, ger_y, ger_y_ids), (en_x, en_x_mask, en_y, en_y_ids) = train_feature
                feature_key_value_pair = {
                    'ger_x': tf.train.Feature(int64_list=tf.train.Int64List(value=ger_x)),
                    'ger_x_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=ger_x_mask)),
                    'ger_y': tf.train.Feature(int64_list=tf.train.Int64List(value=ger_y)),
                    'ger_y_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=ger_y_ids)),
                    'en_x': tf.train.Feature(int64_list=tf.train.Int64List(value=en_x)),
                    'en_x_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=en_x_mask)),
                    'en_y': tf.train.Feature(int64_list=tf.train.Int64List(value=en_y)),
                    'en_y_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=en_y_ids)),
                }
                features = tf.train.Features(feature=feature_key_value_pair)
                example = tf.train.Example(features=features)

                tfwriter.write(example.SerializeToString())
                if ((i + 1) % log_interval) == 0:
                    elapsed = time.time() - start_time
                    print("|T:{}| [{:5d}/{:5d}] | ms/ds_point {:5.2f} |".format(thread_index, i, len(features_dataset),
                                                                                (elapsed * 1000 / log_interval)))
                    start_time = time.time()
        print("[{}] Saved {}".format(thread_index, file_name))

    def write_to_tfrecord_file(self, ds, folder, file_name):
        """
        Write tf record to files.
        Use the number of threads from the class and split the data from the dataset into chunks.
        One chunk will be given to one thread-
        :param ds: Dataset
        :type ds: tf.data.dataset
        :param folder: Folder to write in
        :type folder: str
        :param file_name: Files Names
        :type file_name: str
        """
        Path(folder).mkdir(parents=True, exist_ok=True)
        coord = tf.train.Coordinator()
        processes = []
        for thread_index in range(self.num_threads):
            features_dataset = self.split_ds(thread_index, ds)
            args = (thread_index, features_dataset, folder, file_name)
            p = Process(target=self.write_instances_to_tfrecord, args=args)
            p.start()
            processes.append(p)
        coord.join(processes)