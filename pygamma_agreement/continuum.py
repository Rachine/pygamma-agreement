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
"""
##########
Continuum and corpus
##########
"""
import csv
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import total_ordering
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, List, Union, TYPE_CHECKING, Generator

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.database.util import load_rttm
from sortedcontainers import SortedDict, SortedSet
from typing_extensions import Literal
from .numba_utils import build_A, build_K

from .dissimilarity import AbstractDissimilarity

if TYPE_CHECKING:
    from .alignment import UnitaryAlignment, Alignment, SoftAlignment, WeightedAlignment
    from .sampler import AbstractContinuumSampler, StatisticalContinuumSampler

CHUNK_SIZE = (10**6) // os.cpu_count()

# defining Annotator type
Annotator = str
PivotType = Literal["float_pivot", "int_pivot"]
PrecisionLevel = Literal["high", "medium", "low"]

# percentages for the precision
PRECISION_LEVEL = {
    "high": 0.01,
    "medium": 0.02,
    "low": 0.1
}



@total_ordering
@dataclass(frozen=True, eq=True)
class Unit:
    """
    Represents an annotated unit, e.g., a time segment and (optionally)
    a text annotation. Can be sorted or used in a set. If two units share
    the same time segment, they're sorted alphabetically using their
    annotation. The `None` annotation is first in the "alphabet"

    >>> new_unit = Unit(segment=Segment(17.5, 21.3), annotation='Verb')
    >>> new_unit.segment.start, new_unit.segment.end
    17.5, 21.3
    >>> new_unit.annotation
    'Verb'
    """
    segment: Segment
    annotation: Optional[str] = None

    def __lt__(self, other: 'Unit'):
        if self.segment == other.segment:
            if self.annotation is None:
                return True
            elif other.annotation is None:
                return False
            else:
                return self.annotation < other.annotation
        else:
            return self.segment < other.segment


class Continuum:
    """
    Representation of a continuum, i.e a set of annotated segments by multiple annotators.
    It is implemented as a dictionnary of sets (all sorted) :

    ``{'annotator1': {unit1, ...}, ...}``
    """
    uri: str
    _annotations: SortedDict
    bound_inf: float
    bound_sup: float

    def __init__(self, uri: Optional[str] = None):
        """
        Default constructor.

        Parameters
        ----------
        uri: optional str
            name of annotated resource (e.g. audio or video file)
        """
        self.uri = uri
        # Structure {annotator -> SortedSet}
        self._annotations: SortedDict = SortedDict()
        self._categories: SortedSet = SortedSet()
        self.bound_inf = 0.0
        self.bound_sup = 0.0

        self.best_window_size = np.inf
        # Default best window size. Re-measure it with self.measure_best_window_size

    @classmethod
    def from_csv(cls,
                 path: Union[str, Path],
                 discard_invalid_rows=True,
                 delimiter: str = ","):
        """
        Load annotations from a CSV file , with structure
        annotator, category, segment_start, segment_end.

        .. warning::

            The CSV file mustn't have any header

        Parameters
        ----------
        path: Path or str
            Path to the CSV file storing annotations
        discard_invalid_rows: bool
            If set, every invalid row is ignored when parsing the file.
        delimiter: str
            CSV columns delimiter. Defaults to ','

        Returns
        -------
        Continuum:
            New continuum object loaded from the CSV

        """
        if isinstance(path, str):
            path = Path(path)

        continuum = cls()
        with open(path) as csv_file:
            reader = csv.reader(csv_file, delimiter=delimiter)
            for row in reader:
                seg = Segment(float(row[2]), float(row[3]))
                try:
                    continuum.add(row[0], seg, row[1])
                except ValueError as e:
                    if discard_invalid_rows:
                        print(f"Discarded invalid segment : {str(e)}")
                    else:
                        raise e
        return continuum

    @classmethod
    def from_rttm(cls, path: Union[str, Path]) -> 'Continuum':
        """
        Load annotations from a RTTM file. The file name field will be used
        as an annotation's annotator

        Parameters
        ----------
        path: Path or str
            Path to the CSV file storing annotations

        Returns
        -------
        continuum : Continuum
            New continuum object loaded from the RTTM file
        """
        annotations = load_rttm(str(path))
        continuum = cls()
        for uri, annot in annotations.items():
            continuum.add_annotation(uri, annot)
        return continuum

    def copy_flush(self) -> 'Continuum':
        """
        Returns a copy of the continuum without any annotators/annotations, but with every other information
        """
        continuum = Continuum(self.uri)
        continuum.bound_inf, continuum.bound_sup = self.bound_inf, self.bound_sup
        continuum.best_window_size = self.best_window_size
        return continuum

    def copy(self) -> 'Continuum':
        """
        Makes a copy of the current continuum.

        Returns
        -------
        continuum: Continuum
        """
        continuum = Continuum(self.uri)
        continuum._annotations = deepcopy(self._annotations)
        continuum.bound_inf, continuum.bound_sup = self.bound_inf, self.bound_sup
        continuum.best_window_size = self.best_window_size
        return continuum

    def __bool__(self):
        """Truthiness, basically tests for emptiness

        >>> if continuum:
        ...    # continuum is not empty
        ... else:
        ...    # continuum is empty
        """
        return not all(len(annotations) == 0 for annotations in self._annotations.values())

    def __len__(self):
        return len(self._annotations)

    @property
    def num_units(self) -> int:
        """Total number of units in the continuum."""
        return sum(len(units) for units in self._annotations.values())

    @property
    def categories(self) -> SortedSet:
        """Returns the (alphabetically) sorted set of all the continuum's annotations's categories."""
        return self._categories

    @property
    def category_weights(self) -> SortedDict:
        """
        Returns a dictionary where the keys are the categories in the continuum, and a key's value
        is the proportion of occurrence of the category in the continuum.
        """
        weights = SortedDict()
        nb_units = 0
        for _, unit in self:
            nb_units += 1
            if unit.annotation not in weights:
                weights[unit.annotation] = 1
            else:
                weights[unit.annotation] += 1
        for annotation in weights.keys():
            weights[annotation] /= nb_units
        return weights

    @property
    def bounds(self) -> Tuple[float, float]:
        """Bounds of the continuum. Initially defined as (0, 0),
        they grow as annotations are added."""
        return self.bound_inf, self.bound_sup

    @property
    def num_annotators(self) -> int:
        """Number of annotators"""
        return len(self._annotations)

    @property
    def avg_num_annotations_per_annotator(self) -> float:
        """Average number of annotated segments per annotator"""
        return self.num_units / self.num_annotators

    @property
    def max_num_annotations_per_annotator(self):
        """The maximum number of annotated segments an annotator has
        in this continuum"""
        max_num_annotations_per_annotator = 0
        for annotator in self._annotations:
            max_num_annotations_per_annotator = np.max(
                [max_num_annotations_per_annotator,
                 len(self[annotator])])
        return max_num_annotations_per_annotator

    @property
    def avg_length_unit(self) -> float:
        """Mean of the annotated segments' durations"""
        return sum(unit.segment.duration for _, unit in self) / self.num_units

    def add_annotator(self,  annotator: Annotator):
        """
        Adds the annotator to the set, with no annotated segment. Does nothing if already present.
        """
        if annotator not in self._annotations:
            self._annotations[annotator] = SortedSet()

    def add(self, annotator: Annotator, segment: Segment, annotation: Optional[str] = None):
        """
        Add a segment to the continuum

        Parameters
        ----------
        annotator: Annotator (str)
            The annotator that produced the added annotation
        segment: `pyannote.core.Segment`
            The segment for that annotation
        annotation: optional str
            That segment's annotation, if any.
        """
        if segment.duration == 0.0:
            raise ValueError("Tried adding segment of duration 0.0")

        if annotator not in self._annotations:
            self._annotations[annotator] = SortedSet()
        self._categories.add(annotation)
        self._annotations[annotator].add(Unit(segment, annotation))
        self.bound_inf = min(self.bound_inf, segment.start)
        self.bound_sup = max(self.bound_sup, segment.end)

    def add_annotation(self, annotator: Annotator, annotation: Annotation):
        """
        Add a full pyannote annotation to the continuum.

        Parameters
        ----------
        annotator: Annotator (str)
            A string id for the annotator who produced that annotation.
        annotation: pyannote.core.Annotation
            A pyannote `Annotation` object. If a label is present for a given
            segment, it will be considered as that label's annotation.
        """
        for segment, _, label in annotation.itertracks(yield_label=True):
            self.add(annotator, segment, label)

    def add_timeline(self, annotator: Annotator, timeline: Timeline):
        """
        Add a full pyannote timeline to the continuum.

        Parameters
        ----------
        annotator: Annotator (str)
            A string id for the annotator who produced that timeline.
        timeline: `pyannote.core.Timeline`
            A pyannote `Annotation` object. No annotation will be attached to
            segments.
        """
        for segment in timeline:
            self.add(annotator, segment)

    def add_textgrid(self,
                     annotator: Annotator,
                     tg_path: Union[str, Path],
                     selected_tiers: Optional[List[str]] = None,
                     use_tier_as_annotation: bool = False):
        """
        Add a textgrid file's content to the Continuum

        Parameters
        ----------
        annotator: Annotator (str)
            A string id for the annotator who produced that TextGrid.
        tg_path: `Path` or str
            Path to the textgrid file.
        selected_tiers: optional list of str
            If set, will drop tiers that are not contained in this list.
        use_tier_as_annotation: optional bool
            If True, the annotation for each non-empty interval will be the name
            of its parent Tier.
        """
        from textgrid import TextGrid, IntervalTier
        tg = TextGrid.fromFile(str(tg_path))
        for tier_name in tg.getNames():
            if selected_tiers is not None and tier_name not in selected_tiers:
                continue
            tier: IntervalTier = tg.getFirst(tier_name)
            for interval in tier:
                if not interval.mark:
                    continue

                if use_tier_as_annotation:
                    self.add(annotator,
                             Segment(interval.minTime, interval.maxTime),
                             tier_name)
                else:
                    self.add(annotator,
                             Segment(interval.minTime, interval.maxTime),
                             interval.mark)

    def add_elan(self,
                 annotator: Annotator,
                 eaf_path: Union[str, Path],
                 selected_tiers: Optional[List[str]] = None,
                 use_tier_as_annotation: bool = False):
        """
        Add an Elan (.eaf) file's content to the Continuum

        Parameters
        ----------
        annotator: Annotator (str)
            A string id for the annotator who produced that ELAN file.
        eaf_path: `Path` or str
            Path to the .eaf (ELAN) file.
        selected_tiers: optional list of str
            If set, will drop tiers that are not contained in this list.
        use_tier_as_annotation: optional bool
            If True, the annotation for each non-empty interval will be the name
            of its parent Tier.
        """
        from pympi import Eaf
        eaf = Eaf(eaf_path)
        for tier_name in eaf.get_tier_names():
            if selected_tiers is not None and tier_name not in selected_tiers:
                continue
            for start, end, value in eaf.get_annotation_data_for_tier(tier_name):
                if use_tier_as_annotation:
                    self.add(annotator, Segment(start, end), tier_name)
                else:
                    self.add(annotator, Segment(start, end), value)

    def merge(self, continuum: 'Continuum', in_place: bool = False) -> Optional['Continuum']:
        """
        Merge two Continuua together. Units from the same annotators
        are also merged together (with the usual order of units).

        Parameters
        ----------
        continuum: Continuum
            other continuum to merge into the current one.
        in_place: bool
            If set to true, the merge is done in place, and the current
            continuum (self) is the one being modified. A new continuum
            resulting in the merge is returned otherwise.

        Returns
        -------
        Continuum, optional: Returns the merged copy if in_place is set to True.
        """
        current_cont = self if in_place else self.copy()
        for annotator, unit in continuum:
            current_cont.add(annotator, unit.segment, unit.annotation)
        if not in_place:
            return current_cont

    def __add__(self, other: 'Continuum'):
        """
        Same as a "not-in-place" merge.

        Parameters
        ----------
        other: Continuum
            the continuum to merge into `self`
        """
        return self.merge(other, in_place=False)

    def __getitem__(self, keys: Union[str, Tuple[str, int]]) -> Union[SortedSet, Unit]:
        """Get the set of annotations from an annotator, or a specific annotation.
        (Deep copies are returned to ensure some constraints cannot be violated)

        >>> continuum['Alex']
        SortedSet([Unit(segment=<Segment(2, 9)>, annotation='1'), Unit(segment=<Segment(11, 17)>, ...
        >>> continuum['Alex', 0]
        Unit(segment=<Segment(2, 9)>, annotation='1')

        Parameters
        ----------
        keys: Annotator or Annotator,int


        Raises
        ------
        KeyError
        """
        try:
            if isinstance(keys, str):
                return deepcopy(self._annotations[keys])
            else:
                annotator, idx = keys
                try:
                    return deepcopy(self._annotations[annotator][idx])
                except IndexError:
                    raise IndexError(f'index {idx} of annotations by {annotator} is out of range')
        except KeyError:
            raise KeyError('key must be either Annotator (from the continuum) or (Annotator, int)')

    def __iter__(self) -> Generator[Tuple[Annotator, Unit], None, None]:
        """
        Iterates over (annotator, unit) tuples for every unit in the continuum.
        """
        for annotator, annotations in self._annotations.items():
            for unit in annotations:
                yield annotator, unit

    def iter_annotator(self, annotator: Annotator) -> Generator[Unit, None, None]:
        """
        Iterates over the annotations of the given annotator.

        Raises
        ------
        KeyError
            If the annotators is not on this continuum.
        """
        for unit in self._annotations[annotator]:
            yield unit

    def remove(self, annotator: Annotator, unit: Unit):
        """
        Removes the given unit from the given annotator's annotations.
        Keeps the bounds of the continuum as they are.
        Raises
        ------
        KeyError
            if the unit is not from the annotator's annotations.
        """
        annotations: SortedSet = self._annotations[annotator]
        annotations.remove(unit)

    @property
    def annotators(self) -> SortedSet:
        """Returns a sorted set of the annotators in the Continuum

        >>> self.annotators:
        ... SortedSet(["annotator_a", "annotator_b", "annot_ref"])
        """
        return SortedSet(self._annotations.keys())

    def iterunits(self, annotator: Annotator):
        """Iterate over units from the given annotator
        (in chronological and alphabetical order if annotations are present)

        >>> for unit in self.iterunits("Max"):
        ...     # do something with the unit
        """
        return iter(self._annotations[annotator])

    def get_best_soft_alignment(self,  dissimilarity: AbstractDissimilarity) -> 'SoftAlignment':
        assert len(self.annotators) >= 2 and self, "Disorder cannot be computed with less than two annotators, or " \
                                                   "without annotations."

        sizes = np.empty(self.num_annotators, dtype=np.int32)
        for i, units in enumerate(self._annotations.values()):
            sizes[i] = len(units)

        disorders, possible_unitary_alignments = dissimilarity.valid_alignments(self)
        # Definition of the integer linear program
        n = len(disorders)
        # Constraints matrix ("every unit must appear once and only once")
        A = build_A(possible_unitary_alignments, sizes)

        x = cp.Variable(shape=(n,), boolean=True)
        try:
            import cylp
            cp.Problem(cp.Minimize(disorders.T @ x), [A @ x >= 1]).solve(solver=cp.CBC)
        except (ImportError, cp.SolverError):
            logging.warning("CBC solver not installed. Using GLPK.")
            cp.Problem(cp.Minimize(disorders.T @ x), [A @ x >= 1]).solve(solver=cp.GLPK_MI)
        assert x.value is not None, "The linear solver couldn't find an alignment with minimal disorder " \
                                    "(likely because the amount of possible unitary alignments was too high)"
        # compare with 0.9 as cvxpy returns 1.000 or small values i.e. 10e-14
        chosen_alignments_ids, = np.where(x.value > 0.9)

        chosen_alignments: np.ndarray = possible_unitary_alignments[chosen_alignments_ids]
        alignments_disorders: np.ndarray = disorders[chosen_alignments_ids]

        from .alignment import UnitaryAlignment, SoftAlignment

        set_unitary_alignements = []
        for alignment_id, alignment in enumerate(chosen_alignments):
            u_align_tuple = []
            for annotator_id, unit_id in enumerate(alignment):
                annotator, units = self._annotations.peekitem(annotator_id)
                try:
                    unit = units[unit_id]
                    u_align_tuple.append((annotator, unit))
                except IndexError:  # it's a "null unit"
                    u_align_tuple.append((annotator, None))
            unitary_alignment = UnitaryAlignment(list(u_align_tuple))
            unitary_alignment.disorder = alignments_disorders[alignment_id]
            set_unitary_alignements.append(unitary_alignment)
        return SoftAlignment(set_unitary_alignements,
                             continuum=self,
                             check_validity=True,
                             disorder= np.sum(alignments_disorders) / self.avg_num_annotations_per_annotator)

    def get_first_window(self, dissimilarity: AbstractDissimilarity, w: int = 1) -> Tuple['Continuum', float]:
        """
        Returns a tuple (continuum, x_limit), where :
            - Before x_limit, there are the (w * nb_annotators) leftmost annotations
              of the continuum.
            - After x_limit, there are (approximately) all the annotations from the continuum
              that have a dissimilarity lower than (delta_empty * nb_annotators) with the annotations
              before x_limit.
        """
        # Everything is converted to zippable lists. This is necessary for this method to have
        # a simple and known complexity for choosing the most advantageous window size.
        annotators = list(self.annotators)
        annotations = list(self._annotations.values())
        sizes = list(map(len, annotations))
        indexes = [0] * len(annotators)  # Indexes for "advancing" homogeneously in the continuum

        smallest_unit = Unit(Segment(-np.inf, -np.inf), None)

        window = Continuum()
        for annotator in annotators:
            window.add_annotator(annotator)
        taken_units = 0
        rightmost_unit = smallest_unit
        to_take = min(float(np.sum(sizes)), w * self.num_annotators)
        while taken_units < to_take:  # At least (nb_annotators * w) units

            x_limit = np.inf
            for units, index, size in zip(annotations, indexes, sizes):  # Taking the rightmost unit not already taken
                if index >= size:  # All annotations have been consumed
                    continue
                unit = units[index]
                x_limit = min(x_limit, unit.segment.end)

            for i, (annotator, units, index, size) in enumerate(zip(annotators, annotations, indexes, sizes)):
                if index >= size:  # All annotations have been consumed
                    continue
                unit = units[index]  # Adding the units before x_limit. This will take between 1 and nb_annotator units.
                if unit.segment.end <= x_limit:
                    window.add(annotator, unit.segment, unit.annotation)
                    rightmost_unit = max(unit, rightmost_unit)  # Rightmost taken unit is kept
                    taken_units += 1                            # for selection of additionnal units.
                    indexes[i] += 1

        x_limit = window.bound_sup

        # Now we add the additionnal annotations, "reachable" from those already selected.
        for annotator, units, index, size in zip(annotators, annotations, indexes, sizes):
            while index < size:
                unit = units[index]
                if dissimilarity.d(rightmost_unit, unit) > dissimilarity.delta_empty * self.num_annotators:
                    break
                window.add(annotator, unit.segment, unit.annotation)
                index += 1
        return window, x_limit

    def get_fast_alignment(self, dissimilarity: AbstractDissimilarity, window_size: int) -> 'Alignment':
        """Returns an 'approximation' of the best alignment (Very likely to be the actual best alignment for
         continua with limited overlapping)"""
        from .alignment import Alignment
        copy = self.copy()
        unitary_alignments = []
        disorders = []

        while copy:
            window, x_limit = copy.get_first_window(dissimilarity, window_size)
            # Window contains each annotator's first annotations
            # We retain only the leftmost unitary alignment in the best alignment of the window,
            # as it is the most likely to be in the global best alignment
            best_alignment = window.get_best_alignment(dissimilarity)
            for chosen in best_alignment.take_until_limit(x_limit):
                unitary_alignments.append(chosen)
                disorders.append(chosen.disorder)
                for annotator, unit in chosen.n_tuple:
                    if unit is not None:
                        copy.remove(annotator, unit)  # Now we remove the units from the chosen alignment.
        return Alignment(unitary_alignments,
                         self,
                         check_validity=False,  # Validity has been thoroughly tested
                         disorder=np.sum(disorders) / self.avg_num_annotations_per_annotator)

    def measure_best_window_size(self, dissimilarity: AbstractDissimilarity):
        """
        Sets the best window size for computing the fast-gamma of this continuum, by using the
        sampling the computing complexity function.
        """
        smallest_window, _ = self.get_first_window(dissimilarity, 1)
        smallest_window.get_best_alignment(dissimilarity)

        s = smallest_window.max_num_annotations_per_annotator
        n = int(self.avg_num_annotations_per_annotator)
        p = int(self.num_annotators)

        window_sizes = np.arange(1, max(2, self.max_num_annotations_per_annotator))
        numba_factor = 1/20

        def f(w):
            return (n * p +  # Copying the continuum
                    + ((n - w) * p / 2 + 2 * p + (w + s * p) * p  # getting first window
                        + (n - w) * p / 2 + numba_factor * (w + s) ** p  # getting best alignment
                        + (w + s) * p * np.log2((w + s) * p) + w * p  # getting the w leftmost alignments & adding them
                       ) * (n / w))

        # adding the log factorial corresponds to the emptying-the-continuum-unit-by-unit time.
        logfactorials = np.log2(window_sizes)
        for i in range(1, len(logfactorials)):
            logfactorials[i] += logfactorials[i - 1]

        times = f(window_sizes) + p * logfactorials

        min_index = np.argmin(times)
        if times[min_index] < n * p + numba_factor * n**p:  # Check is fast-gamma is advantageous compared to gamma
            self.best_window_size = window_sizes[min_index]
        else:
            logging.warning("Fast-gamma disadvantageous, using normal gamma.")

    def get_best_alignment(self, dissimilarity: AbstractDissimilarity) -> 'Alignment':
        """
        Returns the best alignment of the continuum for the given dissimilarity. This alignment comes
        with the associated disorder, so you can obtain it in constant time with alignment.disorder.
        Beware that the computational complexity of the algorithm is very high
        :math:`(O(p_1 \\times p_2 \\times ... \\times p_n)` where :math:`p_i` is the number
        of annotations of annotator :math:`i`).

        Parameters
        ----------
        dissimilarity: AbstractDissimilarity
            the dissimilarity that will be used to compute unit-to-unit disorder.
        """
        assert len(self.annotators) >= 2 and self, "Disorder cannot be computed with less than two annotators, or " \
                                                   "without annotations."

        sizes = np.empty(self.num_annotators, dtype=np.int32)
        for i, units in enumerate(self._annotations.values()):
            sizes[i] = len(units)

        disorders, possible_unitary_alignments = dissimilarity.valid_alignments(self)
        # Definition of the integer linear program
        n = len(disorders)
        # Constraints matrix ("every unit must appear once and only once")
        A = build_A(possible_unitary_alignments, sizes)

        x = cp.Variable(shape=(n,), boolean=True)
        try:
            import cylp
            cp.Problem(cp.Minimize(disorders.T @ x), [A @ x == 1]).solve(solver=cp.CBC)
        except (ImportError, cp.SolverError):
            logging.warning("CBC solver not installed. Using GLPK.")
            matmul = A @ x
            cp.Problem(cp.Minimize(disorders.T @ x), [1 <= matmul, matmul <= 1]).solve(solver=cp.GLPK_MI)
        assert x.value is not None, "The linear solver couldn't find an alignment with minimal disorder " \
                                    "(likely because the amount of possible unitary alignments was too high)"
        # compare with 0.9 as cvxpy returns 1.000 or small values i.e. 10e-14
        chosen_alignments_ids, = np.where(x.value > 0.9)

        chosen_alignments: np.ndarray = possible_unitary_alignments[chosen_alignments_ids]
        alignments_disorders: np.ndarray = disorders[chosen_alignments_ids]

        from .alignment import UnitaryAlignment, Alignment

        set_unitary_alignements = []
        for alignment_id, alignment in enumerate(chosen_alignments):
            u_align_tuple = []
            for annotator_id, unit_id in enumerate(alignment):
                annotator, units = self._annotations.peekitem(annotator_id)
                try:
                    unit = units[unit_id]
                    u_align_tuple.append((annotator, unit))
                except IndexError:  # it's a "null unit"
                    u_align_tuple.append((annotator, None))
            unitary_alignment = UnitaryAlignment(list(u_align_tuple))
            unitary_alignment.disorder = alignments_disorders[alignment_id]
            set_unitary_alignements.append(unitary_alignment)
        return Alignment(set_unitary_alignements,
                         continuum=self,
                         # Validity of results from get_best_alignments have been thoroughly tested :
                         check_validity=False,
                         disorder=np.sum(alignments_disorders) / self.avg_num_annotations_per_annotator)

    def compute_gamma(self,
                      dissimilarity: Optional['AbstractDissimilarity'] = None,
                      n_samples: int = 30,
                      precision_level: Optional[Union[float, PrecisionLevel]] = None,
                      ground_truth_annotators: Optional[SortedSet] = None,
                      sampler: 'AbstractContinuumSampler' = None,
                      fast: bool = False,
                      soft: bool = False) -> 'GammaResults':
        """

        Parameters
        ----------
        dissimilarity: AbstractDissimilarity, optional
            dissimilarity instance. Used to compute the disorder between units. If not set, it defaults
            to the combined categorical dissimilarity with parameters taken from the java implementation.
        n_samples: optional int
            number of random continuum sampled from this continuum  used to
            estimate the gamma measure
        precision_level: optional float or "high", "medium", "low"
            error percentage of the gamma estimation. If a literal
            precision level is passed (e.g. "medium"), the corresponding numerical
            value will be used (high: 1%, medium: 2%, low : 5%)
        ground_truth_annotators: SortedSet of str
            if set, the random continuua will only be sampled from these
            annotators. This should be used when you want to compare a prediction
            against some ground truth annotation.
        sampler: AbstractContinuumSampler
            Sampler object, which implements a sampling strategy for creating random continuua used
            to calculate the expected disorder. If not set, defaults to the Statistical continuum sampler
        fast:
            Sets the algorithm to the much faster fast-gamma. It's supposed to be less precise than the "canonical"
            algorithm from Mathet 2015, but usually isn't.
            Performance gains and precision are explained in the Performance section of the documentation.
        """
        from .dissimilarity import CombinedCategoricalDissimilarity
        if dissimilarity is None:
            dissimilarity = CombinedCategoricalDissimilarity()

        if sampler is None:
            from .sampler import StatisticalContinuumSampler
            sampler = StatisticalContinuumSampler()
        sampler.init_sampling(self, ground_truth_annotators)

        job = _compute_best_alignment_job
        if soft:
            job = _compute_soft_alignment_job
        # Multiprocessed computation of sample disorder
        if fast:
            job = _compute_fast_alignment_job
            self.measure_best_window_size(dissimilarity)


        # Multithreaded computation of sample disorder
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as p:
            # Launching jobs
            logging.info(f"Starting computation for the best alignment and a batch of {n_samples} random samples...")
            best_alignment_task = p.submit(job,
                                           *(dissimilarity, self))

            result_pool = [
                # Step one : computing the disorders of a batch of random samples from the continuum (done in parallel)
                p.submit(job,
                         *(dissimilarity, sampler.sample_from_continuum))
                for _ in range(n_samples)
            ]
            chance_best_alignments: List[Alignment] = []
            chance_disorders: List[float] = []

            # Obtaining results
            best_alignment = best_alignment_task.result()
            logging.info("Best alignment obtained")
            for i, result in enumerate(result_pool):
                chance_best_alignments.append(result.result())
                logging.info(f"finished computation of random sample dissimilarity {i + 1}/{n_samples}")
                chance_disorders.append(chance_best_alignments[-1].disorder)
            logging.info("done.")

            if precision_level is not None:
                if isinstance(precision_level, str):
                    precision_level = PRECISION_LEVEL[precision_level]
                assert 0 < precision_level < 1.0
                # If the variation of the disorders of the samples si too high, others are generated.
                # taken from subsection 5.3 of the original paper
                # confidence at 95%, i.e., 1.96
                variation_coeff = np.std(chance_disorders) / np.mean(chance_disorders)
                confidence = 1.96
                required_samples = np.ceil((variation_coeff * confidence / precision_level) ** 2).astype(np.int32)
                if required_samples > n_samples:
                    logging.info(f"Computing second batch of {required_samples - n_samples} "
                                 f"because variation was too high.")
                    result_pool = [
                        p.submit(job,
                                 *(dissimilarity, sampler.sample_from_continuum))
                        for _ in range(required_samples - n_samples)
                    ]
                    for i, result in enumerate(result_pool):
                        chance_best_alignments.append(result.result())
                        logging.info(f"finished computation of additionnal random sample dissimilarity "
                                     f"{i + 1}/{required_samples - n_samples}")
                    logging.info("done.")

        return GammaResults(
            best_alignment=best_alignment,
            chance_alignments=chance_best_alignments,
            precision_level=precision_level,
            dissimilarity=dissimilarity
        )

    def to_csv(self, path: Union[str, Path], delimiter=","):
        if isinstance(path, str):
            path = Path(path)
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=delimiter)
            for annotator, unit in self:
                writer.writerow([annotator, unit.annotation,
                                 unit.segment.start, unit.segment.end])

    def _repr_png_(self):
        """IPython notebook support

        See also
        --------
        :mod:`pygamma_agreement.notebook`
        """

        from .notebook import repr_continuum
        return repr_continuum(self)


@dataclass
class GammaResults:
    """
    Gamma results object. Stores the information about a gamma measure computation,
    used for getting the values of measures from the gamma family (gamma, gamma-cat and gamma-k).
    """
    best_alignment: 'Alignment'
    chance_alignments: List['Alignment']
    dissimilarity: AbstractDissimilarity
    precision_level: Optional[float] = None

    @property
    def n_samples(self):
        """Number of samples used for computation of the expected disorder."""
        return len(self.chance_alignments)

    @property
    def alignments_nb(self):
        """Number of unitary alignments in the best alignment."""
        return len(self.best_alignment.unitary_alignments)

    @property
    def observed_disorder(self) -> float:
        """Returns the disorder of the computed best alignment, i.e, the
        observed disagreement."""
        return self.best_alignment.disorder

    @property
    def expected_disorder(self) -> float:
        """Returns the expected disagreement for computed random samples, i.e.,
        the mean of the sampled continuua's disorders"""
        return float(np.mean([align.disorder for align in self.chance_alignments]))


    @property
    def approx_gamma_range(self):
        """Returns a tuple of the expected boundaries of the computed gamma,
         obtained using the expected disagreement and the precision level"""
        if self.precision_level is None:
            raise ValueError("No precision level has been set, cannot compute"
                             "the gamma boundaries")
        return (1 - self.observed_disorder / (self.expected_disorder *
                (1 - self.precision_level)),
                1 - self.observed_disorder / (self.expected_disorder *
                (1 + self.precision_level)))

    @property
    def gamma(self) -> float:
        """Returns the gamma value"""
        observed_disorder = self.observed_disorder
        if observed_disorder == 0:
            return 1
        return 1 - observed_disorder / self.expected_disorder

    @property
    def gamma_cat(self) -> float:
        """Returns the gamma-cat value"""
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as p:

            observed_disorder_job = p.submit(_compute_gamma_k_job,
                                             *(self.dissimilarity, self.best_alignment, None))

            chance_disorders_jobs = [
                p.submit(_compute_gamma_k_job,
                         *(self.dissimilarity, alignment, None))
                for alignment in self.chance_alignments
            ]
            observed_disorder = observed_disorder_job.result()
            if observed_disorder == 0:
                return 1
            expected_disorder = float(np.mean(np.array([job_res.result() for job_res in chance_disorders_jobs])))

        return 1 - observed_disorder / expected_disorder

    def gamma_k(self, category: str) -> float:
        """Returns the gamma-k value for the given category"""
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as p:
            observed_disorder_job = p.submit(_compute_gamma_k_job,
                                             *(self.dissimilarity, self.best_alignment, category))

            chance_disorders_jobs = [
                p.submit(_compute_gamma_k_job,
                         *(self.dissimilarity, alignment, category))
                for alignment in self.chance_alignments
            ]
            observed_disorder = observed_disorder_job.result()
            if observed_disorder == 0:
                return 1
            expected_disorder = float(np.mean(np.array([job_res.result() for job_res in chance_disorders_jobs])))

        return 1 - observed_disorder / expected_disorder


def _compute_best_alignment_job(dissimilarity: AbstractDissimilarity,
                                continuum: Continuum):
    """
    Function used to launch a multiprocessed job for calculating the best aligment of a continuum
    using the given dissimilarity.
    """
    return continuum.get_best_alignment(dissimilarity)


def _compute_fast_alignment_job(dissimilarity: AbstractDissimilarity,
                                continuum: Continuum):
    """
    Function used to launch a multiprocessed job for calculating an approximation of
    the best aligment of a continuum, using the given dissimilarity.
    """
    if continuum.best_window_size == np.inf:  # window size is set to infinity when normal gamma is better.
        return continuum.get_best_alignment(dissimilarity)
    return continuum.get_fast_alignment(dissimilarity, continuum.best_window_size)

def _compute_soft_alignment_job(dissimilarity: AbstractDissimilarity,
                                continuum: Continuum):
    return continuum.get_best_soft_alignment(dissimilarity)

def _compute_gamma_k_job(dissimilarity: AbstractDissimilarity,
                         alignment: 'Alignment',
                         category: Optional[str]):
    return alignment.gamma_k_disorder(dissimilarity, category)