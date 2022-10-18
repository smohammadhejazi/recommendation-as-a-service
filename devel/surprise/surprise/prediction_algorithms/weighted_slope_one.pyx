"""
the :mod:`weighted_slope_one` module includes the :class:`WeightedSlopeOne` algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np
from six.moves import range
from six import iteritems

from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class WeightedSlopeOne(AlgoBase):
    """A simple yet accurate collaborative filtering algorithm.

    This is a straightforward implementation of the WeightedSlopeOne algorithm
    """

    def __init__(self, virtual_count):

        AlgoBase.__init__(self)
        self.virtual_count = virtual_count

    def fit(self, trainset):
        n_items = trainset.n_items

        # Number of users having rated items i and j: |U_ij|
        cdef np.ndarray[np.int_t, ndim=2] freq
        # Deviation from item i to item j: mean(r_ui - r_uj for u in U_ij)
        cdef np.ndarray[np.double_t, ndim=2] dev

        cdef int u, i, j, r_ui, r_uj

        AlgoBase.fit(self, trainset)

        freq = np.zeros((trainset.n_items, trainset.n_items), np.int32)
        dev = np.zeros((trainset.n_items, trainset.n_items), np.double)

        # Computation of freq and dev arrays.
        for u, u_ratings in iteritems(trainset.ur):
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
                    freq[i, j] += 1
                    dev[i, j] += r_ui - r_uj

        for i in range(n_items):
            dev[i, i] = 0
            for j in range(i + 1, n_items):
                dev[i, j] /= freq[i, j]
                dev[j, i] = -dev[i, j]

        self.freq = freq
        self.dev = dev

        # mean ratings of all users: mu_u
        self.user_mean = [np.mean([r for (_, r) in trainset.ur[u]])
                          for u in trainset.all_users()]

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        # Ri: relevant items for i. This is the set of items j rated by u that
        # also have common users with i (i.e. at least one user has rated both
        # i and j).
        Ri = [j for (j, _) in self.trainset.ur[u] if self.freq[i, j] > 0]
        est = self.user_mean[u]
        if Ri:
            est += sum(self.dev[i, j] for j in Ri) / len(Ri)

        return est
