import sys

from_file = open(sys.argv[1], 'r')
to_file = open(sys.argv[2], 'w')

for line in from_file:
    line = line.strip().split()
    for idx in range(1, len(line)):
        if line[idx] == '?':
            line[idx] = '<unk>'
    to_file.write(' '.join(line)+'\n')
