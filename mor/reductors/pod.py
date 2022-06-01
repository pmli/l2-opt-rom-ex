from pymor.algorithms.pod import pod
from pymor.core.base import BasicObject

from .basic import ABCStationaryRBReductor


class PODReductor(BasicObject):
    """POD reductor.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce.
    """

    def __init__(self, fom):
        self.fom = fom
        self.V = None
        self.s = None
        self.logger.setLevel('INFO')
        self._rb_reductor = None

    def reduce(self, training_set, max_modes, tol, svd_method='qr_svd'):
        """Reduce using strong greedy.

        Parameters
        ----------
        training_set
            The list of parameter values to use in training.
        max_modes
            Maximum order of the ROM.
        tol
            Relative tolerance for POD values.
        svd_method
            SVD method to use in POD (`'method_of_snapshots'` or `'qr_svd'`).
        """
        assert svd_method in ('method_of_snapshots', 'qr_svd')
        X = self.fom.A.source.empty(reserve=len(training_set))
        for mu in training_set:
            X.append(self.fom.solve(mu=mu))
        V, s = pod(X, modes=max_modes, rtol=tol, method=svd_method)
        self.V = V
        self.s = s
        self._rb_reductor = ABCStationaryRBReductor(self.fom, RB=V)
        return self._rb_reductor.reduce()
