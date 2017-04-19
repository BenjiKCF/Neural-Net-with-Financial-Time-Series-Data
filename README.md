# Neural-Network-with-Financial-Time-Series-Data

# Introduction:

Time series Data forms the most paramount part of the quantitative analysis. Today, with the huge amount of data avilable online as well 
as technological advancement, we can analyze large scale data with neural network or aka deep learning, which is not available before. Most importantly, this neural network predicts the future movement of the index and acheives a reasonably well result. In this example, we will predict the SP500.

# Content:

This script can download the data of 7 indexes from online provider, form a pandas DataFrame that is compatitble with the TensorFlow library and finally apply a neural network to it. The codes are indifferent to the original code in google cloud platform because of a) update to avoid using depricated codes and b) simplify it for understanding.

# Result:
Lastest LSTM result (Version3):
will be updated promptly.

Old LSTM result (Version1): 

100epochs, LSTM128 > LSTM128 > 16relu > 1linear 
Train Score: 0.00032 MSE (0.02 RMSE) Test Score: 0.00046 MSE (0.02 RMSE)

100epochs, LSTM80 > LSTM80 > 32relu > 1linear 
Train Score: 0.00039 MSE (0.02 RMSE) Test Score: 0.00258 MSE (0.05 RMSE)

100epochs, LSTM128 > LSTM128 > 64relu > 1linear 
Train Score: 0.00029 MSE (0.02 RMSE) Test Score: 0.00060 MSE (0.02 RMSE)

Old neural network result (Version2):
(Google cloud approach for classification only)
By running 10000 epochs, with 5 neruons in the first hidden layer, 3 neruons in the second hidden layer and 2outputs, it achieves a 0.737 accuracy.
By running 10000 epochs, with 50 neruons in the first hidden layer, 30 neruons in the second hidden layer and 2 outputs, it achieves a 0.806 accuracy.
By running 10000 epochs, with 50 neruons in the first hidden layer, 30 neruons in the second hidden layer and 2 outputs with dropout rate of 0.2, it achieves a 0.77 accuracy.
By running 50000 epochs, with 50 neruons in the first hidden layer, 30 neruons in the second hidden layer and 2 outputs with dropout rate of 0.2, it achieves a 0.815 accuracy.

# How to use it:
The main file should be named as "LSTM_Stock_prediction-date.ipynb"
Run this jupyter notebook with all prerequisite installed. 

# Update:
26/03/2017 First update
1. Recurrent neural network with LSTM are added to the code. 
2. Keras with tensorflow are also implemented. 
3. Tensorboard for neural network visualization are also added to the code.

14/04/2017 Second update
1. Normalized adjusted close price. 
2. A new data downloader has been implemented for simplicity
3. Added more variable to predict the adjusted close price
4. More accurate result, significantly less mean square error
5. Extra visualization for close price
6. Denormalization will be fixed soon
7. Twitter sentiment analysis is currently on testing stage

16/04/2017 Third update
1. Updated denormalization 
2. More test results available

18/04/2017 Fourth update
1. Updated fundamental data from Kaggle for NYSE 

# Future Update:
1. Current day data can be predicted with a live stock data downloader.
2. The current code is to use the open price and high price of the same day to predict the close price. I will try to modify it so that it uses more data to predict the price of stock.
3. Sentiment analysis from tweets and wall street journal will be added too.
4. I will also train the model with 3000 US stocks.
5. New python notebook for Python 3.5 on Windows 10
6. Regularization will be added to avoid overfitting.


# Acknowledgement:
Thanks to google, I have created this neural network with tensorflow, which is an amazing tool that can run on any computer without using the google cloud database. The original tutorial for version 1 is on here, https://www.youtube.com/watch?time_continue=1&v=iBs59GlXhIA and  my LSTM prediction model is based on https://github.com/etai83/lstm_stock_prediction.

I WELCOME you to work together on this interesting project.

