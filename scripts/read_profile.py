import pstats
import sys


def print_stats(filename, sort_by="cumulative", top_n=50):
    stats = pstats.Stats(filename)
    stats.sort_stats(sort_by)
    stats.print_stats(top_n)


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "profile.stats"
    print_stats(filename)
