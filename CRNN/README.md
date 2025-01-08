'pre_process_one_sub.ipynb' will generate dataset for one subject.

'pre_process_all_subs.py' will generate datasets for all subjects. One subject, two files (dataset and label).

'cascade_refactored.py' will train and test model for one subject.

Result

Subject 1:
Epoch 1/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 495s 2s/step - accuracy: 0.5660 - loss: 1.0896 - val_accuracy: 0.9756 - val_loss: 0.0781
Epoch 2/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 477s 1s/step - accuracy: 0.9817 - loss: 0.0594 - val_accuracy: 0.9898 - val_loss: 0.0299
Epoch 3/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9940 - loss: 0.0200 - val_accuracy: 0.9929 - val_loss: 0.0220
Epoch 4/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9961 - loss: 0.0125 - val_accuracy: 0.9930 - val_loss: 0.0228
Epoch 5/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9963 - loss: 0.0117 - val_accuracy: 0.9942 - val_loss: 0.0187
Epoch 6/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 469s 1s/step - accuracy: 0.9975 - loss: 0.0070 - val_accuracy: 0.9954 - val_loss: 0.0166
Epoch 7/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 471s 1s/step - accuracy: 0.9978 - loss: 0.0069 - val_accuracy: 0.9950 - val_loss: 0.0171
Epoch 8/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9976 - loss: 0.0079 - val_accuracy: 0.9961 - val_loss: 0.0134
Epoch 9/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9991 - loss: 0.0031 - val_accuracy: 0.9950 - val_loss: 0.0184
Epoch 10/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 468s 1s/step - accuracy: 0.9971 - loss: 0.0096 - val_accuracy: 0.9958 - val_loss: 0.0170
   1/1012 ━━━━━━━━━━━━━━━━━━━━ 1:10 70ms/step - accuracy: 1.0000 - loss: 4.2415e-   2/1012 ━━━━━━━━━━━━━━━━━━━━ 51s 51ms/step - accuracy: 1.0000 - loss: 7.3013e-0   4/1012 ━━━━━━━━━━━━━━━━━━━━ 48s 49ms/step - accuracy: 1.0000 - loss: 6.6758e-0   6/1012 ━━━━━━━━━━━━━━━━━━━━ 48s 48ms/step - accuracy: 1.0000 - loss: 0.0016   
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 47s 46ms/step - accuracy: 0.9960 - loss: 0.0161
Test Loss: 0.017039882019162178, Test Accuracy: 0.9957688450813293
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 46s 45ms/step  
Recall: [0.9965246  0.99665862 0.99550562 0.99720254 0.99421102 0.99451353]
Precision: [0.99434203 0.99591912 0.99587861 0.99349684 0.99700375 0.99798128]   
F1 Score: [0.99543212 0.99628874 0.99569208 0.99534624 0.99560542 0.99624439]    
AUC: [0.99994086 0.99997615 0.9999848  0.99997419 0.99993752 0.99996583]
Confusion Matrix:
 [[5448    4    7    3    3    2]
 [   8 5369    1    4    2    3]
 [   3    7 5316    7    5    2]
 [   6    1    5 5347    2    1]
 [   9    7    3    9 5324    3]
 [   5    3    6   12    4 5438]]


 Subject 2
 Epoch 1/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 475s 1s/step - accuracy: 0.4155 - loss: 1.4382 - val_accuracy: 0.8123 - val_loss: 0.5551
Epoch 2/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.8260 - loss: 0.5080 - val_accuracy: 0.8862 - val_loss: 0.3400
Epoch 3/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.8967 - loss: 0.3000 - val_accuracy: 0.9045 - val_loss: 0.2846
Epoch 4/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 469s 1s/step - accuracy: 0.9290 - loss: 0.2066 - val_accuracy: 0.9310 - val_loss: 0.2092
Epoch 5/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 469s 1s/step - accuracy: 0.9514 - loss: 0.1423 - val_accuracy: 0.9414 - val_loss: 0.1821
Epoch 6/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 468s 1s/step - accuracy: 0.9639 - loss: 0.1043 - val_accuracy: 0.9475 - val_loss: 0.1632
Epoch 7/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 467s 1s/step - accuracy: 0.9732 - loss: 0.0787 - val_accuracy: 0.9457 - val_loss: 0.1650
Epoch 8/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9779 - loss: 0.0644 - val_accuracy: 0.9528 - val_loss: 0.1499
Epoch 9/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9830 - loss: 0.0505 - val_accuracy: 0.9557 - val_loss: 0.1453
Epoch 10/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 464s 1s/step - accuracy: 0.9830 - loss: 0.0501 - val_accuracy: 0.9608 - val_loss: 0.1310
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 45s 45ms/step - accuracy: 0.9590 - loss: 0.1354 
Test Loss: 0.13095472753047943, Test Accuracy: 0.9608079195022583
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 45s 44ms/step  
Recall: [0.96095647 0.955      0.96265483 0.96490251 0.95783244 0.9635023 ]
Precision: [0.96311552 0.95623957 0.96176579 0.95849474 0.95783244 0.9674255 ]   
F1 Score: [0.96203479 0.95561938 0.96221011 0.96168795 0.95783244 0.96545992]    
AUC: [0.99823111 0.99783481 0.99842044 0.99828696 0.99753092 0.99889567]
Confusion Matrix:
 [[5144   29   49   47   56   28]
 [  45 5157   39   68   57   34]
 [  50   51 5207   28   43   30]
 [  32   54   30 5196   40   33]
 [  35   58   38   46 5179   51]
 [  35   44   51   36   32 5227]]


 Subject 3
uild TensorFlow with the appropriate compiler flags.
Epoch 1/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 473s 1s/step - accuracy: 0.5625 - loss: 1.0957 - val_accuracy: 0.9710 - val_loss: 0.0960
Epoch 2/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9779 - loss: 0.0742 - val_accuracy: 0.9871 - val_loss: 0.0442
Epoch 3/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 469s 1s/step - accuracy: 0.9923 - loss: 0.0268 - val_accuracy: 0.9902 - val_loss: 0.0320
Epoch 4/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9956 - loss: 0.0146 - val_accuracy: 0.9909 - val_loss: 0.0319
Epoch 5/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9958 - loss: 0.0128 - val_accuracy: 0.9924 - val_loss: 0.0269
Epoch 6/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9952 - loss: 0.0146 - val_accuracy: 0.9943 - val_loss: 0.0208
Epoch 7/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9979 - loss: 0.0068 - val_accuracy: 0.9945 - val_loss: 0.0196
Epoch 8/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9983 - loss: 0.0059 - val_accuracy: 0.9933 - val_loss: 0.0264
Epoch 9/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9964 - loss: 0.0102 - val_accuracy: 0.9936 - val_loss: 0.0235
Epoch 10/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 462s 1s/step - accuracy: 0.9979 - loss: 0.0070 - val_accuracy: 0.9952 - val_loss: 0.0159
   1/1012 ━━━━━━━━━━━━━━━━━━━━ 1:08 68ms/step - accuracy: 1.0000 - loss: 2.7636e-   2/1012 ━━━━━━━━━━━━━━━━━━━━ 51s 51ms/step - accuracy: 1.0000 - loss: 2.3871e-0   3/1012 ━━━━━━━━━━━━━━━━━━━━ 58s 58ms/step - accuracy: 1.0000 - loss: 2.7735e-0   5/1012 ━━━━━━━━━━━━━━━━━━━━ 53s 53ms/step - accuracy: 1.0000 - loss: 6.3434e-0   7/1012 ━━━━━━━━━━━━━━━━━━━━ 51s 51ms/step - accuracy: 1.0000 - loss: 7.3010e-0   9/1012 ━━━━━━━━━━━━━━━━━━━━ 49s 49ms/step - accuracy: 1.0000 - loss: 0.0012   
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 45s 45ms/step - accuracy: 0.9953 - loss: 0.0158
Test Loss: 0.01590149663388729, Test Accuracy: 0.995151162147522
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 45s 44ms/step  
Recall: [0.99428361 0.99558092 0.99493528 0.99651312 0.99530781 0.99427728]
Precision: [0.99172338 0.99503128 0.99661781 0.99559956 0.99793    0.99409376]   
F1 Score: [0.99300184 0.99530603 0.99577584 0.99605613 0.99661718 0.99418551]    
AUC: [0.99995489 0.99997631 0.99998183 0.99996705 0.99998363 0.99997325]
Confusion Matrix:
 [[5392    7    4    5    5   10]
 [  11 5407    2    5    2    4]
 [   6    6 5304    7    1    7]
 [   6    2    6 5430    0    5]
 [  10    4    4    1 5303    6]
 [  12    8    2    6    3 5386]]


 Subject 4
 Epoch 1/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 474s 1s/step - accuracy: 0.5494 - loss: 1.1315 - val_accuracy: 0.9670 - val_loss: 0.1084
Epoch 2/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 469s 1s/step - accuracy: 0.9744 - loss: 0.0858 - val_accuracy: 0.9862 - val_loss: 0.0451
Epoch 3/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9903 - loss: 0.0301 - val_accuracy: 0.9892 - val_loss: 0.0355
Epoch 4/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 472s 1s/step - accuracy: 0.9954 - loss: 0.0146 - val_accuracy: 0.9905 - val_loss: 0.0330
Epoch 5/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 471s 1s/step - accuracy: 0.9959 - loss: 0.0125 - val_accuracy: 0.9921 - val_loss: 0.0290
Epoch 6/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9964 - loss: 0.0113 - val_accuracy: 0.9930 - val_loss: 0.0247
Epoch 7/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9974 - loss: 0.0085 - val_accuracy: 0.9928 - val_loss: 0.0269
Epoch 8/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9966 - loss: 0.0099 - val_accuracy: 0.9886 - val_loss: 0.0408
Epoch 9/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 468s 1s/step - accuracy: 0.9956 - loss: 0.0129 - val_accuracy: 0.9945 - val_loss: 0.0205
Epoch 10/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 467s 1s/step - accuracy: 0.9984 - loss: 0.0046 - val_accuracy: 0.9946 - val_loss: 0.0209
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 47s 47ms/step - accuracy: 0.9945 - loss: 0.0203 
Test Loss: 0.020949380472302437, Test Accuracy: 0.9946261644363403
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 48s 47ms/step  
Recall: [0.99391593 0.9928884  0.99516729 0.99665676 0.99441757 0.99474967]
Precision: [0.994833   0.99634035 0.99479747 0.99076809 0.99534364 0.99568318]   
F1 Score: [0.99437425 0.99461138 0.99498235 0.9937037  0.99488039 0.99521621]    
AUC: [0.99994008 0.99990809 0.99997335 0.99995442 0.99994302 0.99994114]
Confusion Matrix:
 [[5391    6    7    9    7    4]
 [   8 5445    6    9   10    6]
 [   7    4 5354   11    3    1]
 [   4    4    2 5366    0    8]
 [   8    4    7    7 5344    4]
 [   1    2    6   14    5 5305]]


 Subject 5
 Epoch 1/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 476s 1s/step - accuracy: 0.4907 - loss: 1.2665 - val_accuracy: 0.9518 - val_loss: 0.1597
Epoch 2/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 472s 1s/step - accuracy: 0.9613 - loss: 0.1269 - val_accuracy: 0.9784 - val_loss: 0.0684
Epoch 3/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 472s 1s/step - accuracy: 0.9860 - loss: 0.0439 - val_accuracy: 0.9833 - val_loss: 0.0523
Epoch 4/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9923 - loss: 0.0237 - val_accuracy: 0.9872 - val_loss: 0.0427
Epoch 5/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 471s 1s/step - accuracy: 0.9933 - loss: 0.0198 - val_accuracy: 0.9887 - val_loss: 0.0381
Epoch 6/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 471s 1s/step - accuracy: 0.9949 - loss: 0.0154 - val_accuracy: 0.9922 - val_loss: 0.0291
Epoch 7/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 471s 1s/step - accuracy: 0.9963 - loss: 0.0104 - val_accuracy: 0.9898 - val_loss: 0.0360
Epoch 8/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 471s 1s/step - accuracy: 0.9956 - loss: 0.0130 - val_accuracy: 0.9901 - val_loss: 0.0357
Epoch 9/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 467s 1s/step - accuracy: 0.9961 - loss: 0.0111 - val_accuracy: 0.9924 - val_loss: 0.0290
Epoch 10/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 468s 1s/step - accuracy: 0.9976 - loss: 0.0067 - val_accuracy: 0.9898 - val_loss: 0.0385
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 47s 46ms/step - accuracy: 0.9902 - loss: 0.0378 
Test Loss: 0.03851592168211937, Test Accuracy: 0.9897773265838623
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 47s 46ms/step  
Recall: [0.9871819  0.99009901 0.99178185 0.99200595 0.99202374 0.98570663]
Precision: [0.99185606 0.98862153 0.99308023 0.99016515 0.98580645 0.98924153]   
F1 Score: [0.98951346 0.98935972 0.99243061 0.9910847  0.98890533 0.98747091]    
AUC: [0.99979202 0.99988181 0.99990003 0.99989918 0.99976307 0.99976588]
Confusion Matrix:
 [[5237    6   11   19   14   18]
 [   7 5300    6    2   25   13]
 [   7    6 5310   10   12    9]
 [  10    8    6 5336    4   15]
 [   6   16    8    8 5348    5]
 [  13   25    6   14   22 5517]]


 Subject 6
 Epoch 1/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 474s 1s/step - accuracy: 0.5418 - loss: 1.1499 - val_accuracy: 0.9671 - val_loss: 0.1102
Epoch 2/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 468s 1s/step - accuracy: 0.9741 - loss: 0.0875 - val_accuracy: 0.9830 - val_loss: 0.0542
Epoch 3/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 467s 1s/step - accuracy: 0.9896 - loss: 0.0324 - val_accuracy: 0.9890 - val_loss: 0.0366
Epoch 4/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 479s 1s/step - accuracy: 0.9951 - loss: 0.0172 - val_accuracy: 0.9888 - val_loss: 0.0419
Epoch 5/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 477s 1s/step - accuracy: 0.9952 - loss: 0.0148 - val_accuracy: 0.9918 - val_loss: 0.0281
Epoch 6/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 477s 1s/step - accuracy: 0.9972 - loss: 0.0097 - val_accuracy: 0.9899 - val_loss: 0.0367
Epoch 7/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 477s 1s/step - accuracy: 0.9953 - loss: 0.0137 - val_accuracy: 0.9924 - val_loss: 0.0272
Epoch 8/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 476s 1s/step - accuracy: 0.9984 - loss: 0.0058 - val_accuracy: 0.9931 - val_loss: 0.0281
Epoch 9/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 481s 1s/step - accuracy: 0.9978 - loss: 0.0064 - val_accuracy: 0.9921 - val_loss: 0.0297
Epoch 10/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 476s 1s/step - accuracy: 0.9968 - loss: 0.0095 - val_accuracy: 0.9923 - val_loss: 0.0281
   1/1012 ━━━━━━━━━━━━━━━━━━━━ 1:29 89ms/step - accuracy: 1.0000 - loss: 2.0244e-   2/1012 ━━━━━━━━━━━━━━━━━━━━ 51s 51ms/step - accuracy: 1.0000 - loss: 4.9947e-0   3/1012 ━━━━━━━━━━━━━━━━━━━━ 52s 52ms/step - accuracy: 1.0000 - loss: 5.3194e-0   5/1012 ━━━━━━━━━━━━━━━━━━━━ 50s 50ms/step - accuracy: 1.0000 - loss: 0.0016   
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 49s 48ms/step - accuracy: 0.9926 - loss: 0.0266
Test Loss: 0.028122087940573692, Test Accuracy: 0.9922789335250854
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 50s 49ms/step  
Recall: [0.99380579 0.99160604 0.99204293 0.99307764 0.99099775 0.9921072 ]
Precision: [0.99001815 0.99494666 0.99185939 0.98974455 0.99416745 0.99301856]   
F1 Score: [0.99190836 0.99327354 0.99195115 0.99140829 0.99258007 0.99256267]    
AUC: [0.99991758 0.99992203 0.99990825 0.9999303  0.99987394 0.99991403]
Confusion Matrix:
 [[5455    6    5   14    5    4]
 [  13 5316    7   13    5    7]
 [  13    5 5361   12    4    9]
 [  12    0    9 5308    9    7]
 [   9    9   12    7 5284   11]
 [   8    7   11    9    8 5405]]


 Subject 7
 Epoch 1/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 475s 1s/step - accuracy: 0.5437 - loss: 1.1621 - val_accuracy: 0.9646 - val_loss: 0.1154
Epoch 2/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 471s 1s/step - accuracy: 0.9718 - loss: 0.0943 - val_accuracy: 0.9851 - val_loss: 0.0494
Epoch 3/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 469s 1s/step - accuracy: 0.9906 - loss: 0.0326 - val_accuracy: 0.9859 - val_loss: 0.0433
Epoch 4/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9925 - loss: 0.0226 - val_accuracy: 0.9893 - val_loss: 0.0349
Epoch 5/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9948 - loss: 0.0171 - val_accuracy: 0.9904 - val_loss: 0.0289
Epoch 6/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 469s 1s/step - accuracy: 0.9953 - loss: 0.0151 - val_accuracy: 0.9913 - val_loss: 0.0298
Epoch 7/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 469s 1s/step - accuracy: 0.9969 - loss: 0.0097 - val_accuracy: 0.9929 - val_loss: 0.0256
Epoch 8/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 470s 1s/step - accuracy: 0.9976 - loss: 0.0078 - val_accuracy: 0.9931 - val_loss: 0.0249
Epoch 9/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 464s 1s/step - accuracy: 0.9967 - loss: 0.0100 - val_accuracy: 0.9916 - val_loss: 0.0291
Epoch 10/10
323/323 ━━━━━━━━━━━━━━━━━━━━ 465s 1s/step - accuracy: 0.9954 - loss: 0.0137 - val_accuracy: 0.9915 - val_loss: 0.0289
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 48s 47ms/step - accuracy: 0.9916 - loss: 0.0280 
Test Loss: 0.028917424380779266, Test Accuracy: 0.9914759397506714
1012/1012 ━━━━━━━━━━━━━━━━━━━━ 47s 46ms/step  
Recall: [0.99631404 0.98877438 0.99205616 0.99238343 0.9906209  0.98868694]
Precision: [0.98739726 0.99224377 0.99168975 0.99091078 0.99080675 0.99589016]   
F1 Score: [0.99183561 0.99050604 0.99187292 0.99164656 0.99071382 0.99227548]    
AUC: [0.99992625 0.9999121  0.99993066 0.99990177 0.99991027 0.99992414]
Confusion Matrix:
 [[5406    4    2    3   10    1]
 [  11 5373   15    9   13   13]
 [   8    7 5370   17    8    3]
 [  16    7    2 5342   13    3]
 [  15   13   13    7 5281    2]
 [  19   11   13   13    5 5331]]