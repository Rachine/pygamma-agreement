"""Tests for the CST & random reference generation"""
from pathlib import Path
from pygamma_agreement.continuum import Continuum
from pygamma_agreement.dissimilarity import CombinedCategoricalDissimilarity, NumericalCategoricalDissimilarity
from pygamma_agreement.cst import CorpusShufflingTool
from pygamma_agreement.sampler import StatisticalContinuumSampler, ShuffleContinuumSampler
import numpy as np
from sortedcontainers import SortedSet


def test_random_reference():
    np.random.seed(4772)
    categories = np.array(["aa", "ab", "ba", "bb"])
    for _ in range(10):  # we do it a certain number of time to be sure no chance happened
        sampler = StatisticalContinuumSampler()

        sampler.init_sampling_custom(annotators=['Martino'], avg_num_units_per_annotator=40,
                                     std_num_units_per_annotator=0,
                                     avg_gap=0, std_gap=5,
                                     avg_duration=10, std_duration=1,
                                     categories=categories)
        continuum = sampler.sample_from_continuum

        assert continuum.categories == SortedSet(categories)
        assert continuum.num_annotators == 1
        assert continuum.annotators[0] == "Martino"

        # this assertion isnt deterministic but probability of passing/failing by chance is like 6.25e-05
        assert abs(continuum.avg_length_unit - 10) < 20

        assert continuum.avg_num_annotations_per_annotator == 40
        assert continuum.num_units == 40


def test_cst_0():
    """
    This test only checks deterministic possibilities. A tool for benchmarking gamma
    using the CST is also provided for more experimental tests.
    """
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/annotation_paul_suzann_alex.csv"))
    categories = continuum.categories
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=3,
                                              beta=1,
                                              cat_dissim=NumericalCategoricalDissimilarity(categories))
    cst_alex = CorpusShufflingTool(0.0, continuum)  # alex is reference

    # A shuffle with magnitude 0.0 just copies the reference.
    shuffled = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                       shift=True,
                                       false_pos=True,
                                       false_neg=True,
                                       cat_shuffle=True,
                                       split=True)
    shuffled_with_reference = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                                      shift=True,
                                                      false_pos=True,
                                                      false_neg=True,
                                                      cat_shuffle=True,
                                                      split=True,
                                                      include_ref=True)
    assert shuffled_with_reference.annotators == SortedSet(("Alex", "Martingale", "Martino"))
    assert shuffled.annotators == SortedSet(("Martingale", "Martino"))
    assert shuffled_with_reference.categories.issubset(categories)
    assert shuffled.categories.issubset(categories)
    assert shuffled_with_reference.compute_gamma(dissim).gamma > 0.9
    assert shuffled.compute_gamma(dissim).gamma > 0.9


def test_cst_1():
    np.random.seed(4772)
    continuum = Continuum.from_csv(Path("tests/data/annotation_paul_suzann_alex.csv"))
    categories = continuum.categories
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=3,
                                              beta=1,
                                              cat_dissim=NumericalCategoricalDissimilarity(categories))
    cst_alex = CorpusShufflingTool(1.0, continuum)  # alex is reference
    shuffled = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                       shift=True,
                                       false_pos=True,
                                       false_neg=True,
                                       cat_shuffle=True,
                                       split=True)
    shuffled_with_reference = cst_alex.corpus_shuffle(["Martino", "Martingale"],
                                                      shift=True,
                                                      false_pos=True,
                                                      false_neg=True,
                                                      cat_shuffle=True,
                                                      split=True,
                                                      include_ref=True)
    assert shuffled.compute_gamma(dissim).gamma < 0.5
    assert shuffled_with_reference.compute_gamma(dissim).gamma < 0.5


def test_cst_cat():
    np.random.seed(5589)
    continuum = Continuum.from_csv(Path("tests/data/annotation_paul_suzann_alex.csv"))
    categories = continuum.categories
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=1,
                                              beta=3,
                                              cat_dissim=NumericalCategoricalDissimilarity(categories))
    cst_alex = CorpusShufflingTool(1.0, continuum)  # alex is reference
    # We test the category shuffle independently for the special options.
    shuffled_cat = cst_alex.corpus_from_reference(["martino", "Martingale", "Martine"])
    cst_alex.category_shuffle(shuffled_cat, prevalence=True)
    # This reference doesn't have enough categories for the gamma to go lower.
    assert shuffled_cat.compute_gamma(dissim).gamma < 0.81

    # Now we generate a reference with A LOT of categories and close segments:
    sampler = StatisticalContinuumSampler()
    sampler.init_sampling_custom(annotators=['Martino'],
                                 avg_num_units_per_annotator=40,
                                 std_num_units_per_annotator=0,
                                 avg_gap=0, std_gap=5,
                                 avg_duration=10, std_duration=1,
                                 categories=np.array([str(i) for i in range(1000)]))
    continuum_martino = sampler.sample_from_continuum
    categories = continuum_martino.categories
    cst_lots_of_cat = CorpusShufflingTool(1.0, continuum_martino)
    shuffled_cat = cst_lots_of_cat.corpus_from_reference(["martino", "Martingale", "Martine"])
    cst_lots_of_cat.category_shuffle(shuffled_cat, prevalence=True)
    dissim = CombinedCategoricalDissimilarity(delta_empty=1,
                                              alpha=1,
                                              beta=3,  # higher beta should make the gamma fall a lot since categories
                                              cat_dissim=NumericalCategoricalDissimilarity(categories))  # are now a mess
    assert shuffled_cat.compute_gamma(dissim).gamma < 0.7


def test_cst_benchmark():

    np.random.seed(4227)
    sampler1 = StatisticalContinuumSampler()
    sampler1.init_sampling_custom(annotators=['Ref'],
                                  avg_num_units_per_annotator=40, std_num_units_per_annotator=0,
                                  avg_duration=20, std_duration=2,
                                  avg_gap=20, std_gap=5,
                                  categories=np.array([str(i) for i in range(10)]))
    reference1 = sampler1.sample_from_continuum
    sampler2 = StatisticalContinuumSampler()
    sampler2.init_sampling_custom(annotators=['Ref'],
                                  avg_num_units_per_annotator=1, std_num_units_per_annotator=0,
                                  avg_duration=6, std_duration=2,
                                  avg_gap=4, std_gap=5,
                                  categories=np.array([str(i) for i in range(1)]))
    reference2 = sampler2.sample_from_continuum

    nb_gammas_per_magnitude = 3
    nb_annotators = 3

    for reference in (reference1, reference2):
        dissim = CombinedCategoricalDissimilarity(alpha=3, beta=0, delta_empty=1.0)

        cst = CorpusShufflingTool(1.0, reference)

        for m in (1, np.random.uniform(0.1, 0.9), 0):
            for _ in range(nb_gammas_per_magnitude):
                cst.magnitude = m
                cont_cst = cst.corpus_shuffle(nb_annotators, shift=True, split=True, false_neg=True, cat_shuffle=True)
                gamma_results = cont_cst.compute_gamma(dissim,
                                                       sampler=ShuffleContinuumSampler(),
                                                       fast=True)
                gamma = gamma_results.gamma
                gamma_cat = gamma_results.gamma_cat
