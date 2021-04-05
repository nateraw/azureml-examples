import os
from argparse import ArgumentParser
from pathlib import Path

from pprint import pprint


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--message', type=str, default='Hello, world!')
    args = parser.parse_args()
    
    logdir = Path('./logs')
    logdir.mkdir(exist_ok=True, parents=True)
    outfile = logdir / 'message.txt'
    outfile.write_text(args.message)
    print(f"Message: {args.message}")
    print('-'*40)
    print()
    pprint(dict(os.environ))
    print('-'*40)
    print(f"Current Directory: {Path.cwd()}")
    print('-'*40)
