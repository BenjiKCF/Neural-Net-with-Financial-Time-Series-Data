# Neural-Network-with-Financial-Time-Series-Data

# Introduction:

Time-series Data forms the most paramount part of the quantitative analysis. Today, with the huge amount of data available online as well as technological advancement, we can analyze large scale data with recurrent neural network easily. Most importantly, this neural network predicts the future movement of the index and achieves a reasonably well result (0.01RMSE on Test set data). In this example, we will predict the SP500.

# Content:

This script downloads the data of stock or indexes from the online provider, form a pandas DataFrame that contains open, high, low, close and is compatible with the TensorFlow library and Keras. Finally, apply a neural network to it. Finally, a visualized graph will be presented to compare the accuracy of it.t.

# Result:
Lastest LSTM model result for 7 years of testing data that has not been trained:

![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/result1.png)

Train Score: 0.00002 MSE (0.00 RMSE)

Test Score: 0.00012 MSE (0.01 RMSE)

# Hyperparameter
Currently, I am testing for the best hyperparameter for this model.
For dropout, the result is shown as below.
![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/dropout.png)

For epochs, the result is probably less than a 100 epochs, more test will be conducted
![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/epochs.png)

# How to use it:
The main file should be named as "LSTM_Stock_prediction-date.ipynb"
Run this jupyter notebook with all prerequisite installed. 

# Update:
26/03/2017 First update
1. Recurrent neural network with LSTM is added to the code. 
2. Keras with tensorflow is also implemented. 
3. Tensorboard for neural network visualization is also added to the code.

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

19/04/2017 Fifth update
1. Supporting Python 3.5 on Windows 10
2. Significant improvement in accuracy

29/04/2017 Sixth update
1. ^GSPC Data since 1970 has been added, more training data, higher accuracy
2. 7 years of test data 
3. Object oriented programming
4. Fine tune for dropout will be added soon


# Future Update:
1. Current day data can be predicted with a live stock data downloader.
2. The current code is to use the open price and high price of the same day to predict the close price. I will try to modify it so that it uses more data to predict the price of stock.
3. Sentiment analysis from tweets and wall street journal will be added too.
4. I will also train the model with 3000 US stocks.
5. New python notebook for Python 3.5 on Windows 10
6. Regularization will be added to avoid overfitting.
7. Quantopian Zipline will be used for backtesting
8. Auto selection for the best hyperparameter


# Acknowledgement:
Thanks to google, I have created this neural network with tensorflow, which is an amazing tool that can run on any computer without using the google cloud database. The original tutorial for version 1 is on here, https://www.youtube.com/watch?time_continue=1&v=iBs59GlXhIA and my LSTM prediction model is based on https://github.com/etai83/lstm_stock_prediction.

I WELCOME you to work together on this interesting project and improve the model.

