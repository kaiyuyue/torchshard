def check_divisibility(numerator, denominator):
    assert numerator % denominator == 0, \
        '{} is not divisible by {}'.format(numerator, denominator)

def divide(numerator, denominator):
    check_divisibility(numerator, denominator)
    return numerator // denominator

class VocabUtility:
    """
    Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indecies in [fist, last).
    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                  rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size)
