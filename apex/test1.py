import sys
import argparse
import time

def main(args):
    print("Hello")
    print(sys.argv)
    print(args)
    print(args.local_rank)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=1, type=int)
    args = parser.parse_args()
    main(args)
