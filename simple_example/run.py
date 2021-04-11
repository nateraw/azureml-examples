from argparse import ArgumentParser
from pathlib import Path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--message', type=str, default="Hello, world!")
    args = parser.parse_args()
    
    logdir = Path('./logs')
    logdir.mkdir(exist_ok=True, parents=True)
    outfile_path = logdir / 'message.txt'
    outfile_path.write_text(args.message)
