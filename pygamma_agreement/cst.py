# The MIT License (MIT)

# Copyright (c) 2020-2021 CoML

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Rachid RIAD, Hadrien TITEUX, Léopold FAVRE

import logging
from typing import Union, Iterable, Callable
import numpy as np
import numpy.random
from pyannote.core import Segment
from sortedcontainers import SortedSet

from .continuum import Annotator, Continuum


class CorpusShufflingTool:
    """
    Corpus shuffling tool as detailed in section 6.3 of the gamma paper
    (https://www.aclweb.org/anthology/J15-3003.pdf#page=30).
    """
    SHIFT_FACTOR = 2
    SPLIT_FACTOR = 2.5
    FALSE_POS_FACTOR = 1

    def __init__(self,
                 magnitude: float,
                 reference_continuum: Continuum,
                 categories: Iterable[str] = None):
        """
        Parameters
        ----------
        magnitude:
            magnitude m of the cst (cf gamma paper)
        reference_continuum:
            this continuum will serve as reference for the tweaks made by the corpus shuffling tool.
        categories:
            this is used to consider additionnal categories when shuffling the corpus, in the eventuality that the
            reference continuum does not contain any unit of a possible category.
        """
        self.magnitude: float = magnitude
        reference_annotators = reference_continuum.annotators
        if len(reference_annotators) > 1:
            logging.warning("Warning : a reference continuum with multiple annotators was given to the CST, so "
                            "its first annotator in alphabetical order will be used as reference.")
        self._reference_annotator: Annotator = reference_annotators[0]
        self._reference_continuum: Continuum = reference_continuum
        self._categories: SortedSet = self._reference_continuum.categories
        if categories is not None:
            for category in categories:
                self._categories.add(category)

    def corpus_from_reference(self, new_annotators: Union[int, Iterable[Annotator]]):
        # TODO: add docstring
        continuum = Continuum()
        continuum._categories = self._reference_continuum.categories
        continuum.bound_inf, continuum.bound_sup = self._reference_continuum.bounds
        if isinstance(new_annotators, int):
            new_annotators = [f"annotator_{i}" for i in range(new_annotators)]
        for unit in self._reference_continuum.iter_annotator(self._reference_annotator):
            for new_annotator in new_annotators:
                continuum.add(new_annotator,
                              Segment(unit.segment.start, unit.segment.end),
                              unit.annotation)
        return continuum

    def shift_shuffle(self, continuum: Continuum) -> None:
        """
        Tweaks the given continuum by shifting the ends of each segment, with uniformly distributed values
        of bounds proportionnal to the magnitude of the CST and the length of the segment.
        """
        shift_max = self.magnitude * self.SHIFT_FACTOR * \
            self._reference_continuum.avg_length_unit
        for annotator in continuum.annotators:
            for unit in continuum[annotator]:
                continuum.remove(annotator, unit)
                start_seg, end_seg = 0.0, 0.0
                while start_seg >= end_seg:
                    start_seg = unit.segment.start + np.random.uniform(-1, 1) * shift_max
                    end_seg = unit.segment.end + np.random.uniform(-1, 1) * shift_max
                continuum.add(annotator, Segment(start_seg, end_seg), unit.annotation)

    def false_neg_shuffle(self, continuum: Continuum) -> None:
        """
        Tweaks the continuum by randomly removing units ("false negatives").
        Every unit (for each annotator) have a probability equal to the magnitude of being removed.
        If this probability is one, a single random unit (for each annotator) will be left alone.
        """
        for annotator in continuum.annotators:
            security = np.random.choice(continuum._annotations[annotator])
            # security : if an annotator doesnt have any annotations gamma cant be computed.
            for unit in list(continuum[annotator]):
                if np.random.random() < self.magnitude:
                    continuum.remove(annotator, unit)
            if len(continuum._annotations[annotator]) == 0:
                continuum.add(annotator, security.segment, security.annotation)

    def false_pos_shuffle(self, continuum: Continuum) -> None:
        """
        Tweaks the continuum by randomly adding "false positive" units.
        The number of added units per annotator is constant & proportionnal to the magnitude of the CST.
        The chosen category is random and depends on the probability of occurence of the category in the reference.
        The length of the segment is random (normal distribution) based on the average and standard deviation
        of those of the reference.
        """
        ref_units = self._reference_continuum[self._reference_annotator]
        avg_dur = np.average([unit.segment.end - unit.segment.start for unit in ref_units])
        var_dur = np.std([unit.segment.end - unit.segment.start for unit in ref_units])
        category_weights = self._reference_continuum.category_weights
        bounds_inf, bounds_sup = self._reference_continuum.bound_inf, self._reference_continuum.bound_sup
        for annotator in continuum.annotators:
            for _ in range(int(self.magnitude * self.FALSE_POS_FACTOR * len(self._reference_continuum))):
                # a random unit is generated from a (all random) central point, duration, and category
                category = np.random.choice(category_weights.keys(), p=category_weights.values())
                center = np.random.uniform(bounds_inf, bounds_sup)
                duration = abs(np.random.normal(avg_dur, var_dur))
                continuum.add(annotator,
                              Segment(center - duration / 2, center + duration / 2),
                              annotation=category)

    def category_shuffle(self, continuum: Continuum,
                         overlapping_fun: Callable[[str, str], float] = None,
                         prevalence: bool = False):
        """
        Shuffles the categories of the annotations in the given continuum using the process described in
        section 3.3.5 of https://hal.archives-ouvertes.fr/hal-00769639/.
        Parameters
        ----------
        overlapping_fun:
            gives the "categorical distance" between two annotations, which is taken into account when provided.
            (the lower the distance between categories, the higher the chance one will be changed into the other).
        prevalence:
            specify whether or not to consider the proportion of presence of each category in the reference.
        """
        category_weights = self._reference_continuum.category_weights
        # matrix "A"
        nb_categories = len(category_weights)
        prob_matrix = np.eye(nb_categories)
        # matrix "B or C"
        if prevalence:
            sec_matrix = np.ones(nb_categories) / nb_categories
        else:
            sec_matrix = np.array([list(category_weights.values())] * nb_categories)

        categories = list(category_weights.keys())
        if overlapping_fun is None:
            # this formula was deduced from the graphs.
            prob_matrix = prob_matrix * (1 - self.magnitude ** 2) + sec_matrix * self.magnitude ** 2
        else:
            overlapping_matrix = np.zeros((len(categories), len(categories)))
            for id1, cat1 in enumerate(categories):
                sum_line = 0
                for id2, cat2 in enumerate(categories):
                    elem = overlapping_fun(cat1, cat2)
                    sum_line += elem
                    overlapping_matrix[id1, id2] = elem
                overlapping_matrix[id1] /= sum_line
            # this formula was also deduced from the graphs.
            prob_matrix = (prob_matrix * (1 - self.magnitude)
                           + sec_matrix * self.magnitude ** 3
                           + overlapping_matrix * (self.magnitude - self.magnitude ** 3)
                           )
        for annotator in continuum.annotators:
            for unit in list(continuum[annotator]):
                continuum.remove(annotator, unit)
                try:
                    new_category = np.random.choice(categories, p=prob_matrix[category_weights.index(unit.annotation)])
                except ValueError as e:
                    raise e
                continuum.add(annotator, Segment(unit.segment.start, unit.segment.end), new_category)
                del unit

    def splits_shuffle(self, continuum: Continuum):
        """
        Tweak the continuum by randomly splitting segments.
        Number of splits per annotator is constant & proportionnal to the magnitude of the CST
        and the number of units in the reference.
        A splitted segment can be re-splitted.
        """
        for _ in range(int(self.magnitude *
                           self.SPLIT_FACTOR *
                           self._reference_continuum.avg_num_annotations_per_annotator)):
            for annotator in continuum.annotators:
                units = continuum._annotations[annotator]
                to_split = units.pop(numpy.random.randint(0, len(units)))
                security = (to_split.segment.end - to_split.segment.start) * 0.01
                cut = numpy.random.uniform(to_split.segment.start + security, to_split.segment.end)


                try:
                    continuum.add(annotator, Segment(cut, to_split.segment.end), to_split.annotation)
                    continuum.add(annotator, Segment(to_split.segment.start, cut), to_split.annotation)
                except ValueError:
                    continuum.add(annotator, to_split.segment, to_split.annotation)
                    continuum.add(annotator, to_split.segment, to_split.annotation)


    def corpus_shuffle(self,
                       annotators: Union[int, Iterable[str]],
                       shift: bool = False,
                       false_pos: bool = False,
                       false_neg: bool = False,
                       split: bool = False,
                       cat_shuffle: bool = False,
                       include_ref: bool = False
                       ) -> Continuum:
        """
        Generates a new shuffled corpus with the provided (or generated) reference annotation set,
        using the method described in 6.3 of the gamma paper, https://www.aclweb.org/anthology/J15-3003.pdf#page=30
        (and missing elements described in another article : https://hal.archives-ouvertes.fr/hal-00769639/).
        """
        continuum = self.corpus_from_reference(annotators)
        if shift:
            self.shift_shuffle(continuum)
        if false_pos:
            self.false_pos_shuffle(continuum)
        if false_neg:
            self.false_neg_shuffle(continuum)
        if cat_shuffle:
            self.category_shuffle(continuum)
        if split:
            self.splits_shuffle(continuum)
        if include_ref:
            assert self._reference_annotator not in continuum.annotators, \
                "Reference annotator can't be included as " \
                "an annotator with the same name is in the " \
                "generated corpus."
            for unit in self._reference_continuum[next(iter(self._reference_continuum.annotators))]:
                continuum.add(self._reference_annotator, unit.segment, unit.annotation)
        return continuum
