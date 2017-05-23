import os
os.system('ffm-train  -v 10 -l 0.0005 -t 20 -s 16 -r 0.1 fea_for_ffm.txt ffm_train.model')
os.system('ffm-predict fea_for_ffm_test.txt ffm_train.model ffm_output.txt')