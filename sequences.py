class Sequence:
    """
    Represents a sequence. This class is meant to be inherited.
    """
    
    def __init__(self):
        pass

    def get_element(self, i):
        """
        Gets the i-th element in the sqeuence.

        Args:
            i: Index of pattern to grab. Should be i >= 0.
        Returns:
            The i-th element in the sequence.
        """
        return i

    def get_sample(self, start, stop):
        """
        Creates a sample of the sequence.

        Args:
            start: First index in sequence to begin sample at, inclusive.
            stop: Index to end sequence sample at, exclusive.
        Returns:
            The sample as a tuple, from [start, stop) of the sequence.
        """
        sample = [self.get_element(i) for i in range(start, stop)]
        return tuple(sample)

class Squares(Sequence):
    """
    Represents a sequence of squares, such that the i-th element is i^2.
    """
    
    def get_element(self, i):
        return i**2

class Fibonacci(Sequence):
    """
    The Fibonacci sequence.
    """

    def __init__(self):
        self.cache = [1, 1]

    def get_element(self, i):
        if i >= len(self.cache):
            for j in range(len(self.cache), i + 1):
                self.cache.append(self.cache[j - 2] + self.cache[j - 1])
        return self.cache[i]

class Triangular(Sequence):
    """
    The triangular numbers sequence.
    """

    def get_element(self, i):
        return int((i * (i + 1)) / 2)

class Pentagonal(Sequence):
    """
    The pentagonal numbers sequence.
    """

    def get_element(self, i):
        return int((3*i**2 + 5*i + 2) / 2)

class Hexagonal(Sequence):
    """
    The hexagonal numbers sequence.
    """

    def get_element(self, i):
        return 2 * i**2 + 3 * i + 1
