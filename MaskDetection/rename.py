
## 1. Prepare the dataset

import os
#Rename the pictures from 1 to ... 
class BatchRename():

    def rename(self):
        path = "/Users/chengming/Desktop/datasets/mldata/mask"
        filelist = os.listdir(path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(path), item)
                dst = os.path.join(os.path.abspath(path), ''+str(i)+'.jpg')
                try:
                    os.rename(src, dst)
                    i += 1
                except:
                    continue
# The main function
if __name__=='__main__':
    demo = BatchRename()
    demo.rename()