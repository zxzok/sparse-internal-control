"""
MCQA feedback with the configured local-estimator microbatch size.
"""
from __future__ import annotations
import argparse
import sys
from sparse_control.runners.implementations.mcqa_microbatch import install_microbatched_estimator

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--microbatch-size', type=int, default=2)
    known, remaining = parser.parse_known_args()
    install_microbatched_estimator(known.microbatch_size)
    sys.argv = [sys.argv[0], *remaining]
    from sparse_control.runners.implementations.mcqa_feedback import main as submitted_main
    submitted_main()
if __name__ == '__main__':
    main()
