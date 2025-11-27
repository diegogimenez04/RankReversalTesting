import skcriteria as skc
from skcriteria.pipelines import mkpipe
from skcriteria.preprocessing.scalers import SumScaler, VectorScaler, MinMaxScaler
from skcriteria.preprocessing.invert_objectives import InvertMinimize
from skcriteria.agg.simple import WeightedSumModel
from skcriteria.ranksrev import RankInvariantChecker, RankTransitivityChecker


dm = skc.datasets.load_van2021evaluation(windows_size=7)

dmaker = skc.pipeline.mkpipe(
        InvertMinimize(),
        SumScaler(target="weights"),
        MinMaxScaler(target="matrix"),
        WeightedSumModel(),
    )

res = RankTransitivityChecker(dmaker).evaluate(dm=dm)

print(res.ranks)