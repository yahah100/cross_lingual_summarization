import re


class CnnDailyMailData:
    """
    CNN/Daily Mail class for reading and tokenizing the data
    """
    def __init__(self, tokenize_helper, parallel=True) -> None:
        """
        Init
        :param tokenize_helper: Tokenize Helper which is able to tokenize parallel
        :type tokenize_helper: tokenize_helper.TokenizeHelper
        :param parallel: boolean if parallel tokenization will be used
        :type parallel: bool
        """
        super().__init__()
        self.tokenize_helper = tokenize_helper
        self.parallel = parallel

    @staticmethod
    def transform(x):
        x = re.sub("'(.*)'", r"\1", x)
        return x # + "</s>"

    @staticmethod
    def transfrom_and_remove_index(x):
        x = " ".join(x.split("; ")[1:])
        x = re.sub("'(.*)'", r"\1", x)
        return x # + "</s>"

    def get_cdm_english_data(self, name):
        article_path = "../../data/cnn_daily_mail/{}/article".format(name)
        highlights_path = "../../data/cnn_daily_mail/{}/highlights".format(name)

        articles = [self.transform(x.rstrip()) for x in open(article_path).readlines()]
        highlights = [self.transform(x.rstrip()) for x in open(highlights_path).readlines()]
        assert len(articles) == len(highlights), "cdm articles:{} highlights:{}".format(len(articles), len(highlights))
        return articles, highlights

    def get_cdm_german_data(self, name):
        article_path = "../../data/cnn_daily_mail/{}/articles_german".format(name)
        highlights_path = "../../data/cnn_daily_mail/{}/highlights_german".format(name)

        articles = [self.transfrom_and_remove_index(x.rstrip()) for x in open(article_path).readlines()]
        highlights = [self.transfrom_and_remove_index(x.rstrip()) for x in open(highlights_path).readlines()]
        return articles, highlights

    def get_tokenized_multilingual_ds(self, name):
        """
        Get tokenized multilingual dataset from text files
        :param name: "train" "test" "val"
        :type name: str
        :return: ger_tokenized, en_tokenized
        :rtype: tuple of lists
        """
        cdm_ger_articles, cdm_ger_highlights = self.get_cdm_german_data(name)
        cdm_en_articles, cdm_en_highlights = self.get_cdm_english_data(name)

        if self.parallel:
            ger_tokenized = self.tokenize_helper.get_parallel_tokenized_ds(cdm_ger_articles, cdm_ger_highlights)
            en_tokenized = self.tokenize_helper.get_parallel_tokenized_ds(cdm_en_articles, cdm_en_highlights)
        else:
            ger_tokenized = self.tokenize_helper.get_tokenized_ds(cdm_ger_articles, cdm_ger_highlights)
            en_tokenized = self.tokenize_helper.get_tokenized_ds(cdm_en_articles, cdm_en_highlights)

        return ger_tokenized, en_tokenized
