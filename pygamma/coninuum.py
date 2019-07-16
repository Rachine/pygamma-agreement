#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2019 CNRS

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
# Rachid RIAD
"""
##########
Annotation
##########

"""

import itertools
import numpy as np

from pyannote.core import Segment, Timeline, Annotation


class Continuum(object):
    """Continuum
    Parameters
    ----------
    uri : string, optional
        name of annotated resource (e.g. audio or video file)
    modality : string, optional
        name of annotated modality
    Returns
    -------
    continuum : Continuum
        New continuum
    """

    def __init__(self, uri=None, modality=None):

        super(Continuum, self).__init__()

        self._uri = uri
        self.modality = modality

        # sorted dictionary
        # keys: annotators
        # values: {annotator: annotations} dictionary
        self._annotators = SortedDict()

        # timeline meant to store all annotated segments
        self._timeline = None
        self._timelineNeedsUpdate = True

    def _get_uri(self):
        return self._uri

    def __len__(self):
        """Number of annotators
        >>> len(continuum)  # continuum contains 3 annotators
        3
        """
        return len(self._annotators)

    def __bool__(self):
        """Emptiness
        >>> if continuum:
        ...    # continuum is empty
        ... else:
        ...    # continuum is not empty
        """
        return len(self._annotators) > 0

    def itersegments(self):
        """Iterate over segments (in chronological order)
        >>> for segment in annotation.itersegments():
        ...     # do something with the segment
        See also
        --------
        :class:`pyannote.core.Segment` describes how segments are sorted.
        """
        return iter(self._annotators)
