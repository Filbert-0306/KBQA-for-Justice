# CSV-->TXT文件(将demo里面的'副本_Stu_Exe.csv'文件)

fr = open('../data/justice_raw_data/xsbh10&16_template_train.csv', 'rt')
fw = open('../data/justice/xsbh10&16_template_train.txt', 'w+')

ls = []

for line in fr:
    line = line.replace('\n', '')  # 删除每行后面的换行符
    line = line.split(',')  # 将每行数据以逗号切割成单个字符
    ls.append(line)  # 将单个字符追加到列表ls中

for row in ls:
    fw.write('\t'.join(row) + '\n') # tab键分隔

fr.close()
fw.close()