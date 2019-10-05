import argparse
import sys
import operator
import conlleval

parser = argparse.ArgumentParser()

parser.add_argument('--TF', type=str, help="training F1 as weight")
parser.add_argument('--datapath', type=str, help="data path")
parser.add_argument('--true_datapath', type=str, default="/home/ron_data/ouyu/data/conll2003/ner/ner-mturk/pruned_ground_truth.txt", help="data path of ground truth")

args = parser.parse_args()

tf = [float(n) for n in args.TF.split()]
print(tf)
a_num = len(tf)
print(a_num)
infiles = [open(args.datapath+'_'+str(i), 'r') for i in range(a_num)]
infiles = [f.readlines() for f in infiles]
print([len(f) for f in infiles])
tufile = open(args.true_datapath, 'r')
tufile = tufile.readlines()
print(args.datapath, len(infiles[0]))
print(args.true_datapath, len(tufile))
outfile = open(args.datapath, 'w')

data=[]
for i in range(len(infiles[0])):
    line = infiles[0][i].strip()
    if len(line) == 0:
        outfile.write('\n')
    else:
        #w, t = tufile[i].strip().split()
        w, t, _ = line.split()
        ps = {}
        for j in range(a_num):
            p = infiles[j][i].strip().split()[-1]
            if p not in ps:
                ps[p] = 0
            ps[p] += tf[j]
        sorted_ps = sorted(ps.items(), key=operator.itemgetter(1))
        p = sorted_ps[-1][0]
        outfile.write(w+' '+t+' '+p+'\n')
        data.append(w+' '+t+' '+p)

conlleval.evaluate(data)
