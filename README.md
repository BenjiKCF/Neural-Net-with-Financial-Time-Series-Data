# Neural-Network-with-Financial-Time-Series-Data

# Introduction:

Time-series Data forms the most paramount part of the quantitative analysis. Today, with the huge amount of data available online as well as technological advancement, we can analyze large scale data with recurrent neural network easily. Most importantly, this neural network predicts the future movement of the index and achieves a reasonably well result (0.01RMSE on Test set data). In this example, we will predict the SP500.

# Content:

This script downloads the data of stock or indexes from the online provider, form a pandas DataFrame that contains open, high, low, close and is compatible with the TensorFlow library and Keras. The prediction is based on the open, high, low in the same day to predict the adjusted close price in the very last minute or hour. (This method maybe inappropriate because the high and low data may not be available until the very end of the day, new version of prediction will be provided to address this problem.) Finally, apply a neural network to it. Finally, a visualized graph will be presented to compare the accuracy of it.

# How it works:

The concept of this model is mimicking technical analysis which uses the past price as well as the current high, low in the same day to predict the closing price.

# Disadvantage of this model:
I believe in the  efficient market hypothesis (EMH) that price cannot be predicted based on previous price. And this model is breaking the rule of it since it uses high, low data in the same day, which should be a future data. I wish to improve it if I can get my hands on hourly market data from 9:30 to 16:00. 

# Future improvement:
1. Hourly data for this model.
2. Uses more fundamental data to predict the price of stock.
3. Sentiment analysis
4. Train the model with 3000 US stocks.
5. Deep Q learning for portfolio optimization and risk
6. Regularization will be added to avoid overfitting.
7. Quantopian Zipline will be used for backtesting
8. Auto selection for the best hyperparameter



# Result:
Lastest LSTM model result for 7 years of testing data that has not been trained:

![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/result2.png)

Train Score: 0.00006 MSE (0.01 RMSE)

Test Score: 0.00029 MSE (0.02 RMSE)

# Hyperparameter
Currently, I am testing for the best hyperparameter for this model.

For dropout, the result is shown as below.

![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/dropout.png)

For epochs less than 100 would be ideal.

![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/epochs.png)

![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/epochs2.png)

For number of neurons, [256, 256, 32, 1] and [512, 512, 32, 1] would be ideal for this model.

![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/neurons.png)

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


# Acknowledgement:
Thanks to google, I have created this neural network with tensorflow, which is an amazing tool that can run on any computer without using the google cloud database. The original tutorial for version 1 is on here, https://www.youtube.com/watch?time_continue=1&v=iBs59GlXhIA and my LSTM prediction model is based on https://github.com/etai83/lstm_stock_prediction.

# References:
Bernal, A., Fok, S., & Pidaparthi, R. (2012). Financial Market Time Series Prediction with Recurrent Neural Networks.

G. E. P. Box, G. M. Jenkins, and G. C. Reinsel. Time series analysis: forecasting and control, volume
734. Wiley, 2011.

H. Jaeger. The "echo state" approach to analysing and training recurrent neural networks-with an
erratum note. Tecnical report GMD report, 148, 2001.

H. Jaeger. Tutorial on training recurrent neural networks, covering BPPT, RTRL, EKF and the" echo
state network" approach. GMD-Forschungszentrum Informationstechnik, 2002.

Wolfgang Maass, Thomas Natschlager, and Henry Markram. Real-time computing without stable states:
A new framework for neural computation based on perturbations. Neural Computation, 14(11):2531–
2560, November 2002.

I WELCOME you to work together on this interesting project and improve the model.

