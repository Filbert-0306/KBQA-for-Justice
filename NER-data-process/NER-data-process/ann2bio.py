import glob

def bratann2BIO_format(text, ann_str, fstream):
  """
  text:巴中兴文置信号
  使用brat 等标注工具，标注完导出为 brat-ann 格式如下：
  T1 (这里是\t)行政区划 0 2  (这里是\t)巴中
  T2 行政区划 2 4  兴文
  T3 核心词 4 6 置信
  运行ann2bio.py，把许多条.ann转化成train.txt那样的格式，并且要保证有对应的一条一条的txt
  """
  ann_list = ann_str.strip().split('\n')
  label = ['O' for _ in range(len(text))]
  for i,line in enumerate(ann_list):
    try:
      T,typ,word = line.strip().split('\t')
      t,s,e = typ.split()
      s,e = int(s),int(e)
      label[s] = 'B-'+t
      while s < e-1:
        s += 1
        label[s] = 'I-'+t
    except:
      continue
   
  for t,l in zip(list(text),label):
    line = ' '.join([t,l]) # 可以使用空格代替
    fstream.write(line)
    fstream.write('\n')
  fstream.write('\n')


def gen_NER_training_data():
  # 设置标注文件所在文件夹目录
  root_dir = '../../build_kg/prepare_data/qa_data/tag_txt_data/outputs'
  # 设置训练样本输出文件路径
  stream = open('../../build_kg/prepare_data/qa_data/train_ann_txt/xsbh_16_ner.txt', 'a+', encoding='utf8')
  # ann:E:\工作空间\NER-data-process\data\biaozhushuju\outputs\10000.ann
  # txt:E:\工作空间\NER-data-process\data\biaozhushuju\10000.txt

  file_list = glob.glob(root_dir+'/*.ann')
  for ann_path in file_list:
    ann_path = ann_path.replace('\\','/')
    txt_path = ann_path.replace('/outputs','').replace('ann','txt')
    try:
      ft = open(txt_path, 'r', encoding='utf8')
      text = ft.read().strip()
      ft.close()
      fa = open(ann_path,'r',encoding='utf8')
      ann = fa.read().strip()
      fa.close()
      if ann == '':
        continue
      bratann2BIO_format(text, ann, stream)
    except Exception as e:
      print(ann_path,e)
       
  stream.close()

if __name__ == '__main__':
  gen_NER_training_data()