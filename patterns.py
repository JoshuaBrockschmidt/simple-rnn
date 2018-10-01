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
        sample = (self.get_element(i) for i in range(start, stop))
