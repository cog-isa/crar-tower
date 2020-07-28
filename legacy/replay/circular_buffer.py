import sys


class CircularBuffer(object):
    def __init__(self, size, elemShape=(), extension=0.1, dtype="float32"):
        self._size = size
        self._data = np.zeros((int(size + extension * size),) + elemShape, dtype=dtype)
        self._trueSize = self._data.shape[0]
        self._lb = 0
        self._ub = size
        self._cur = 0
        self.dtype = dtype

    def append(self, obj):
        if self._cur > self._size:  # > instead of >=
            self._lb += 1
            self._ub += 1

        if self._ub >= self._trueSize:
            # Rolling array without copying whole array (for memory constraints)
            # basic command: self._data[0:self._size-1] = self._data[self._lb:] OR NEW self._data[0:self._size] = self._data[self._lb-1:]
            n_splits = 10
            for i in range(n_splits):
                self._data[i * (self._size) // n_splits:(i + 1) * (self._size) // n_splits] = self._data[
                                                                                              (self._lb - 1) + i * (
                                                                                                  self._size) // n_splits:(
                                                                                                                                      self._lb - 1) + (
                                                                                                                                      i + 1) * (
                                                                                                                              self._size) // n_splits]
            self._lb = 0
            self._ub = self._size
            self._cur = self._size  # OLD self._size - 1

        self._data[self._cur] = obj
        self._cur += 1

    def __getitem__(self, i):
        return self._data[self._lb + i]

    def getSliceBySeq(self, seq):
        return self._data[seq + self._lb]

    def getSlice(self, start, end=sys.maxsize):
        if end == sys.maxsize:
            return self._data[self._lb + start:self._cur]
        else:
            return self._data[self._lb + start:self._lb + end]

    def getLowerBound(self):
        return self._lb

    def getUpperBound(self):
        return self._ub

    def getIndex(self):
        return self._cur

    def getTrueSize(self):
        return self._trueSize
