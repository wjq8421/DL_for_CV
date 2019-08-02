import argparse

parser = argparse.ArgumentParser()
parser.description='喂我两个数字，我就吐出他们的积'
parser.add_argument("ParA", help="我是A",type=int)
parser.add_argument("ParB", help="我是B",type=int)
args = parser.parse_args()
print('嗯，好吃！积是', args.ParA*args.ParB)