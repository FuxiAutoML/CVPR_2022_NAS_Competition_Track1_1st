import os, sys, time
import json

arch_acc = open(r'./channel_sample.txt', 'r', encoding='utf-8')

output1 = {}
output2 = {}
i = 1
for line in arch_acc:
    key  = f'arch{i}'
    i    += 1
    line = line.strip().split(' ')
    arch = line[0]
    acc1 = float(line[1])
    acc2 = float(line[2])
    val1 = {'acc': acc1, 'arch': arch}
    val2 = {'acc': acc2, 'arch': arch}
    output1[key] = val1
    output2[key] = val2

output1 = json.dumps(output1)
output2 = json.dumps(output2)

with open(r'./result_acc1.json', 'w', encoding='utf-8') as res:
    res.write(output1)
    
with open(r'./result_acc2.json', 'w', encoding='utf-8') as res:
    res.write(output2)