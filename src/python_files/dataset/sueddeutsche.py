import re


class SueddeutscheData:
    """
    Sueddeutsche class for reading and tokenizing the data
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
        x = " ".join(x.split("; ")[1:])
        x = re.sub("'(.*)'", r"\1", x)
        return x  # + "</s>"

    def get_sueddeutsche_data(self, name, language, switch_test):
        if language == "en":
            highlights_path = "../../data/sueddeutsche_old/highlights_{}_{}_cleaned".format(language, name)
            if switch_test:
                article_path = "../../data/sueddeutsche_old/articles_{}_{}_switched".format(language, name)
            else:
                article_path = "../../data/sueddeutsche_old/articles_{}_{}_cleaned".format(language, name)
        else:
            highlights_path = "../../data/sueddeutsche_old/highlights_{}_{}".format(language, name)
            if switch_test:
                article_path = "../../data/sueddeutsche_old/articles_{}_{}_switched".format(language, name)
            else:
                article_path = "../../data/sueddeutsche_old/articles_{}_{}".format(language, name)

        articles = [self.transform(x.rstrip()) for x in open(article_path).readlines()]
        highlights = [self.transform(x.rstrip()) for x in open(highlights_path).readlines()]
        assert len(articles) == len(highlights), "sueddeutsche_old articles:{} highlights:{}".format(len(articles),
                                                                                                 len(highlights))

        return articles, highlights

    def get_tokenized_multilingual_ds(self, name, switch_test=False):
        """
        Get tokenized multilingual dataset from text files
        :param name: "train" "test" "val"
        :type name: str
        :param switch_test: boolean to indicate if the special switch data should be used
        :type switch_test: bool
        :return: ger_tokenized, en_tokenized
        :rtype: tuple of lists
        """
        sd_ger_articles, sd_ger_highlights = self.get_sueddeutsche_data(name, "de", switch_test)
        sd_en_articles, sd_en_highlights = self.get_sueddeutsche_data(name, "en", switch_test)

        if self.parallel:
            ger_tokenized = self.tokenize_helper.get_parallel_tokenized_ds(sd_ger_articles, sd_ger_highlights)
            en_tokenized = self.tokenize_helper.get_parallel_tokenized_ds(sd_en_articles, sd_en_highlights)
        else:
            ger_tokenized = self.tokenize_helper.get_tokenized_ds(sd_ger_articles, sd_ger_highlights)
            en_tokenized = self.tokenize_helper.get_tokenized_ds(sd_en_articles, sd_en_highlights)

        return ger_tokenized, en_tokenized

