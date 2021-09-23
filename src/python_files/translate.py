import torch
from pathlib import Path
from spacy.lang.en import English
from spacy.lang.de import German
import time
import re


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyNlp:
    """
    Spacy NLP class for sentence splitting
    """
    def __init__(self, input_language) -> None:
        super().__init__()

        if input_language == "en":
            self.nlp = English()
        else:
            self.nlp = German()
        self.nlp.add_pipe('sentencizer')

    def split_in_sentences(self, text):
        doc = self.nlp(text)
        return [str(sent).strip() for sent in doc.sents]


def get_model(language_to_translate):
    """
    Get pretrained translation model from pytorch hub
    :param language_to_translate:
    :type language_to_translate: str
    :return: pretrained model
    :rtype:
    """
    model_name = ""
    if language_to_translate == "en":
        model_name = "transformer.wmt19.{}.single_model".format("en-de")
    else:
        model_name = "transformer.wmt19.{}.single_model".format("de-en")

    # Load an En-De Transformer model trained on WMT'19 data:
    en2de = torch.hub.load('pytorch/fairseq', model_name, tokenizer='moses', bpe='fastbpe')

    # Access the underlying TransformerModel
    assert isinstance(en2de.models[0], torch.nn.Module)

    # to gpu
    return en2de.to(device)


def remove_index(str_in):
    return "; ".join(str_in.split("; ")[1:])


def read_files(name, path="../../data/sueddeutsche_old/"):
    """
    Read articles and highlights
    :param name: "train" "test" "val"
    :type name: str
    :param path: path of the files
    :type path: str
    :return: articles and highlights
    :rtype: dict
    """
    articles = []
    highlights = []
    article_path = path + "articles_de_{}".format(name)
    with open(article_path, "r") as article_file:
        for line in article_file.readlines():
            articles.append(remove_index(line))

    highlight_path = path + "highlights_de_{}".format(name)
    with open(highlight_path, "r") as highlight_file:
        for line in highlight_file:
            highlights.append(remove_index(line))
    return {'articles': articles, 'highlights': highlights}


class MyDataset(torch.utils.data.Dataset):
    """
    Torch Dataset class
    """
    def __init__(self, articles, nlp):
        self.x = articles
        self.nlp = nlp

    def __getitem__(self, index):
        sentences = self.nlp.split_in_sentences(self.x[index])
        ret_x = []
        for i, sent in enumerate(sentences):
            ret_x.append(sent[:1024])

        return ret_x

    @staticmethod
    def transfrom(x):
        x = x.lower()
        x = re.sub("'(.*)'", r"\1", x)
        return x

    def __len__(self):
        return len(self.x)


class FileWriter:
    """
    Class Filewriter to get the index of the last translation and append lines to this file
    """
    def __init__(self, ds_name, name, path="../../data/sueddeutsche_old/"):
        self.path = path + name + "_en_{}".format(ds_name)
        self.file = Path(self.path).open("a")

    def write_translated(self, i, list_str):
        """
        write line of translated data to file
        :param i: index
        :type i: int
        :param list_str: sentences
        :type list_str: list
        """
        result = str(i) + "; "
        for item_str in list_str:
            result += item_str + " "
        self.file.write(result.replace("\n", " ") + "\n")
        self.file.flush()

    def get_last_index(self):
        """
        Get last index
        :return: index
        :rtype: int
        """
        with open(self.path) as fileObj:
            ret_list = list(fileObj)
            if len(ret_list) > 0:
                return int(ret_list[-1].split(";")[0])
            else:
                return 0


def translate(ds, ds_name, name, log_interval=1000):
    """
    translate
    :param ds: Dataset
    :type ds: MyDataset
    :param ds_name: "train" "test" "val"
    :type ds_name: str
    :param name: "articles" or "highlight"
    :type name: str
    :param log_interval: interval to print log msg
    :type log_interval: int
    """
    len_ds = len(ds)
    model = get_model("de")
    file_writer = FileWriter(ds_name, name)
    first_index = file_writer.get_last_index()
    start_time = time.time()
    for i in range(first_index, len_ds):
        predictions = model.translate(ds[i])
        file_writer.write_translated(i, predictions)
        if ((i+1) % log_interval) == 0:
            elapsed = time.time() - start_time
            print("| [{:5d}/{:5d}] | ms/ds_point {:5.2f} |".format(i, len(ds), (elapsed * 1000 / log_interval)))
            start_time = time.time()


def main():
    nlp = MyNlp("de")
    ds_name = "train"
    raw_ds = read_files(ds_name)
    for summary_part in ['articles', 'highlights']:
        file_writer = FileWriter(ds_name, summary_part)
        file_writer.get_last_index()

        ds = MyDataset(raw_ds[summary_part], nlp)
        translate(ds, ds_name, summary_part)


if __name__ == '__main__':
    main()
