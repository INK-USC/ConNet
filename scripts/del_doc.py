infile="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/train"
outfile="/home/ron_data/ouyu/data/conll2003/ner/raw_bio/train_clean"

inf = open(infile, 'r')
inf = inf.readlines()
out = open(outfile, 'w')

i = 0
while i < len(inf):
    line = inf[i].strip()
    if len(line) > 0 and line.split()[0] in ['-DOCSTART-']:
        i += 2
    else:
        out.write(inf[i])
        i += 1
