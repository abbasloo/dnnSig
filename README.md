# dnnComm: unsupervised deep learning for wireless communication signals with different modulations and SNR's

Detail:

- I/Q trajectory space is embeded to a new dimention (encoded by color in trajectory space and its histogram is the signature) by an autoencoder which considers time-lags for signal evolution in trajectory space (Markov modeling).

![alt text](https://github.com/abbasloo/dnnComm/blob/master/result.png)

- An experiment using embeded data and SVM has been done to see the AUC curve and if the new dimention introduces strong discriminative feature.

![alt text](https://github.com/abbasloo/dnnComm/blob/master/AUC.png)
    
Dataset:
  
    https://www.deepsig.io/datasets/

Based on the following project:

    https://github.com/msmbuilder/vde
