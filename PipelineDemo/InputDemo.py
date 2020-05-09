import argparse

parser = argparse.ArgumentParser(description='PrepareInputExample', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--input_file", type=str, default='/home/mohamedt/scientific_data/data/test')
parser.add_argument("--output_file", type=str, default='/home/mohamedt/scientific_data/data/demo.input')
parser.add_argument('--paper_id', type=int, default=1)

args = parser.parse_args()

file_name=args.input_file
file_output=args.output_file

file=open(file_name,'r')

lines=file.readlines()
line=lines[args.paper_id]

file=open(file_output,'w')
file.write(line)
file.close()

