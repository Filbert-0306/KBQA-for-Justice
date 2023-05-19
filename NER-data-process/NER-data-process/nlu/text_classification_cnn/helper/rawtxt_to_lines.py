import os
import sys

"""
	将raw_txt_data里的txt文件转换成一条一条的，存储在tag_txt_data，方便BIO打标
"""

raw_data_path = '../data/xsbh10_template_test.csv'

save_folder = '../data/justice'
if not os.path.exists(save_folder):
	os.makedirs(save_folder)

with open(raw_data_path,'r',encoding='gbk') as f:
	for i,line in enumerate(f.readlines()):
		line = line.strip()

		with open(os.path.join(save_folder,'特征_%s.txt' % str(i)),'w',encoding='utf-8') as fo:	# 可以换成10000+i之类的，方便区分
			fo.write(line)
