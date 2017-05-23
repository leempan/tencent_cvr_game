import numpy as np
import pandas as pd

newfile = open('merged_new_fea_test.csv','w')
fealist = ['none_id_fea_test.txt','ad_id_fea_test.txt','adv_id_fea_test.txt','app_id_fea_test.txt','cam_id_fea_test.txt','cre_id_fea_test.txt','pos_id_fea_test.txt','site_id_fea_test.txt','user_id_fea_test.txt']

files = []
for ffile in fealist:
	files.append(open(ffile))
ind = 0
while True:
	end = False
	A = []
	for file in files:
		line = file.readline()
		if not line:
			end = True
			break
		line = line.split(',')
		numbers = [int(float(d)) for d in line]
		A.extend(numbers)
	if end:
		break
	print ind
	# print A,len(A)
	if ind == 0:
		s=''
		for i in range(len(A)-1):
	   		s=s+'f'+str(i)+','
		s = s+str(len(A)-1)+'\n'
		newfile.write(s) 
	s =str(A)
	newfile.write(s[1:len(s)-1]+'\n')
	ind = ind+1
	
for f in files:
	f.close()
newfile.close()