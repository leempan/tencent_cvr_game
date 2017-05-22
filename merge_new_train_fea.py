import numpy as np
import pandas as pd

newfile = open('merged_new_fea_train.csv','w')
train_data_raw = '/home/administrator/Limingpan/pre/merged_train.csv'
train_raw = pd.read_csv(train_data_raw,header=0)
label = train_raw['label'].as_matrix()
fealist = ['none_id_fea.txt','ad_id_fea.txt','adv_id_fea.txt','app_id_fea.txt','cam_id_fea.txt','cre_id_fea.txt','pos_id_fea.txt','site_id_fea.txt','user_id_fea.txt']

files = []
for ffile in fealist:
	files.append(open(ffile))
ind = 0

while True:
	end = False
	if ind >= len(label):
		break
	A = [label[ind]]
	for file in files:
		line = file.readline()
		line = line.split(',')
		numbers = [int(float(d)) for d in line]
		A.extend(numbers)
	print ind
	if ind == 0:
	   s='label,'
           for i in range(len(A)-1):
		s=s+'f'+str(i)+','
	   newfile.write(s) 				
	s =str(A)
	newfile.write(s[1:len(s)-1]+'\n')
	ind = ind+1
for f in files:
	f.close()
newfile.close()
