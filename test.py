import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--input", help="datapath to input csv file")
parser.add_argument("--output", help="datapath to output csv file")


args = parser.parse_args()

for arg in vars(args):
	print(arg, getattr(args, arg))
