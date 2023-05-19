import os
import sys

raw_data_path = './data/raw_data.txt'

save_folder = './data/biaozhushuju'
if not os.path.exists(save_folder):
	os.makedirs(save_folder)

with open(raw_data_path,'r',encoding='utf8') as f:
	for i,line in enumerate(f.readlines()):
		line = line.strip()

		with open(os.path.join(save_folder,'%s.txt' % str(10000+i)),'w',encoding='utf8') as fo:
			fo.write(line)
