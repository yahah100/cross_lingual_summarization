from os import listdir
import os
import time
import pickle

class SummaryGenerator:
    """
    Generate Summaries
    """
    def __init__(self, tokenizer, language_token_order, save_parts, epoch=None) -> None:
        """
        Init
        :param tokenizer: transformers Tokenizer
        :type tokenizer:  transformers Tokenizer
        :param language_token_order: order to generate the summary in the right ordxer
        :type language_token_order: list
        :param save_parts: Boolean if
        :type save_parts: bool
        :param epoch: Epochs
        :type epoch: int
        """
        super().__init__()
        self.save_parts = save_parts
        self.tokenizer = tokenizer
        self.language_token_order = language_token_order
        self.epoch = epoch

    @staticmethod
    def get_all_language_combinations(ds):
        for i in range(1, 5):
            yield ds[(i-1)*4], ds[i*4-3], ds[i*4-2], ds[i*4-1]


    @staticmethod
    def get_key(elem):
        key = int(elem.split(".")[0].split("-")[1])
        return key

    def get_last_index(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        list_dir = list(listdir(path))
        list_dir.sort(key=self.get_key)
        if len(list_dir) == 0:
            last_index = -1
        else:
            last_index = int(list_dir[-1].split(".")[0].split("-")[1])
            print(last_index)
        return last_index


    def save_predictions(self, i, predictions, path):
        if self.epoch is None:
            with open(path + "results-{}.pickle".format(i), "wb") as file:
                pickle.dump(predictions, file)
        else:
            with open(path + "epoch-{}-results-{}.pickle".format(self.epoch, i), "wb") as file:
                pickle.dump(predictions, file)

    def generate_summaries(self, model, test_ds, path, use_break_point=True, break_point=20):
        """

        :param model: Model T5 for generation
        :type model: tf model
        :param test_ds: Test dataset
        :type test_ds: tf.data.dataset
        :param path: Path of result folder
        :type path: str
        :param use_break_point: Use break point to break summarizing
        :type use_break_point: bool
        :param break_point: point to break loop
        :type break_point: int
        :return: Predictions
        :rtype: list
        """
        predictions = []
        start_time = time.time()
        log_interval = 10
        t_i = 0
        last_index = self.get_last_index(path)
        for i, ds_item in enumerate(test_ds):
            t_i = i
            if i > last_index:
                for j, (input_ids, input_mask, y, y_ids) in enumerate(self.get_all_language_combinations(ds_item)):
                    summaries = model.generate(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        num_beams=4,
                        length_penalty=0.6,
                        early_stopping=True,
                        max_length=150
                    )

                    articles = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                                input_ids]

                    pred = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                            summaries]
                    real = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in y]

                    for pred_sent, real_sent, article in zip(pred, real, articles):
                        predictions.append({
                            'language_tag': self.language_token_order[j],
                            'input_data':article,
                            'real_data':real_sent,
                            'pred_data':pred_sent
                        })

                if (i % log_interval) == 0 and i != 0:
                    elapsed = (time.time() - start_time)
                    print("[{}]: time generating {} batches: {}".format(i, log_interval, elapsed))
                    if self.save_parts:
                        self.save_predictions(i, predictions, path)
                        predictions = []
                    start_time = time.time()

                if use_break_point:
                    if i > break_point:
                        # otherwise it will take ages
                        break

        self.save_predictions(t_i, predictions, path)
        return predictions
