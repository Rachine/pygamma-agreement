import pygamma_agreement as pa
from pyannote.core import Segment
import numpy as np


def test_soft_alignment_check():
    unit_martin, unit_martino = pa.Unit(Segment(0, 10), "a"), pa.Unit(Segment(5, 15), "b")
    continuum = pa.Continuum()

    continuum.add("Martin", Segment(0, 10), "a")
    continuum.add("Martino", Segment(5, 15), "b")

    unitary_alignment_1 = pa.UnitaryAlignment([("Martin", unit_martin),
                                               ("Martino", None)])
    unitary_alignment_2 = pa.UnitaryAlignment([("Martin", unit_martin),
                                               ("Martino", unit_martino)])
    unitary_alignment_3 = pa.UnitaryAlignment([("Martin", None),
                                               ("Martino", unit_martino)])

    alignment = pa.alignment.SoftAlignment([unitary_alignment_3, unitary_alignment_1, unitary_alignment_2],
                                           continuum,
                                           check_validity=True)


def test_soft_gamma_comp():
    n = 30
    p = 3

    sampler = pa.StatisticalContinuumSampler()
    sampler.init_sampling_custom(annotators=['Ref'],
                                 avg_num_units_per_annotator=n, std_num_units_per_annotator=0,
                                 avg_duration=5, std_duration=2,
                                 avg_gap=3, std_gap=2,
                                 categories=np.array([str(i) for i in range(4)]))
    # Compilation
    continuum = sampler.sample_from_continuum
    cst = pa.CorpusShufflingTool(0.3, continuum)
    continuum = cst.corpus_shuffle([f"annotator_{i}" for i in range(p)], split=True)

    # continuum = pa.Continuum.from_csv("tests/data/AlexPaulSuzan.csv")
    # continuum._annotations.pop("Alex")

    dissim = pa.CombinedCategoricalDissimilarity(delta_empty=1,
                                                 alpha=1,
                                                 beta=1)

    seed = np.random.randint(0, 10000)

    np.random.seed(seed)
    soft_gamma = continuum.compute_gamma(dissim, soft=True)

    np.random.seed(seed)
    gamma = continuum.compute_gamma(dissim)
    print(gamma.gamma)
    print(soft_gamma.gamma)

    assert soft_gamma.gamma >= 1.5 * gamma.gamma

