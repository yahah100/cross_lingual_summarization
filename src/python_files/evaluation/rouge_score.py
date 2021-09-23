from rouge_score import rouge_scorer
from rouge_score import scoring

class RougeScore:
    '''
    mostly from https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/evaluation/metrics.py
    '''

    def __init__(self, score_keys=None) -> None:
        super().__init__()
        if score_keys is None:
            self.score_keys = ["rouge1", "rouge2", "rougeLsum"]

        self.scorer = rouge_scorer.RougeScorer(self.score_keys)
        self.aggregator = scoring.BootstrapAggregator()

    @staticmethod
    def prepare_summary(summary):
        # Make sure the summary is not bytes-type
        # Add newlines between sentences so that rougeLsum is computed correctly.
        summary = summary.replace(" . ", " .\n")
        return summary

    def __call__(self, target, prediction):
        """
        Computes rouge score
        :param target: Target sequence
        :type target: str
        :param prediction: Predicted sequence
        :type prediction: str
        """
        target = self.prepare_summary(target)
        prediction = self.prepare_summary(prediction)

        self.aggregator.add_scores(self.scorer.score(target=target, prediction=prediction))

    def result(self):
        """
        Get results
        :return: Result dict with multiple ROUGE score keys
        :rtype: dict
        """
        result = self.aggregator.aggregate()

        for key in self.score_keys:
            score_text = "%s = %.2f, 95%% confidence [%.2f, %.2f]" % (
                key,
                result[key].mid.fmeasure * 100,
                result[key].low.fmeasure * 100,
                result[key].high.fmeasure * 100
            )
            print(score_text)

        return {key: result[key].mid.fmeasure * 100 for key in self.score_keys}