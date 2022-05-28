# CPS-3320-MaskChecking

Describe:
Due to the epidemic, masks have become a necessity for People's Daily travel. For example, in Wenzhou Kean University, students are also required to wear masks when eating in the canteen to avoid the risk of cross-infection. And sometimes even though there are staff checking whether to wear masks at the entrance of the canteen, some people will try their best to avoid being checked. So, in order to strictly require students to wear masks, we want to design a program to check whether they wear masks or not.
The final program output should look something like this. Photos of people will be automatically identified by the machine as wearing masks and marked on the photos.


Run the program:
There are four parts of code, of which "rename.py" and "divideDatasets.py" are used to rename and separate datasets. We mainly run the full code of "MaskDetection.py" or a code of "camera.py" which only has camera detection. "MaskDetection.py" is a complete mask detection program, including data preprocessing and model training, as well as processes such as drawing loss functions. During this process, a "mask_and_unmask_small_1.h5" file storage model is generated. We have generated one such file, so you can use it to run the code of "camera.py" directly. without the need for a long training process.
