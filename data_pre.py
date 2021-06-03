import csv
import os

from tqdm import tqdm

f = open('财经.csv','w',encoding='utf-8') #这里可以通过修改文件名来选择不同的文件，生成不同的cvs文件
csv_writer = csv.writer(f)
csv_writer.writerow(["file_name","class_name","text"])
file_path = '分类样本集'
class_name = '财经'
rootdir = '分类样本集/财经'
list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
for i in tqdm(range(0,len(list))):
    path = os.path.join(rootdir, list[i])
    with open(path, "r") as df:
        text=df.read().replace('\n','').replace('　','').strip()
    csv_writer.writerow([list[i],class_name,text])
    df.close()
f.close()
