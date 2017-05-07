# Neural-Network-with-Financial-Time-Series-Data

# Introduction:

Time-series Data forms the most paramount part of the quantitative analysis. Today, with the huge amount of data available online as well as technological advancement, we can analyze time-series data with recurrent neural network easily. This neural network predicts the future movement of the index and achieves a reasonably well result.

# Content:

This script downloads the data of stock or indexes from the online provider, form a pandas DataFrame that contains open, high, low, close and is compatible with the TensorFlow library and Keras. Finally, apply a neural network to the data and create a visualized graph. Hyperparameter is also provided at the end for choosing optimal hyperparameter.

# Versions
After receiving the feedback that stock price should not be predicted with the data from the same date. From now on, there will be 2 versions with the similar method to predict the stock price.

1. Prediction with 21 previous days and today open high low (Original) (Regression)
2. Prediction with 22 previous days (Modified) (Regression)
3. Prediction with 22 previous days (Modified) (Classification)

With version 2, it can avoid using "future" data for predition.
With version 3, it can classify gain and loss of today.

# How it works:

The concept of this model is mimicking technical analysis which uses the past prices to predict the closing price.

# Disadvantage of this model:
I believe in the efficient market hypothesis (EMH) that price cannot be predicted based on previous price. This model attempts to understand the market sentiment behind price trends rather than analyzing a security's fundamental attributes. In order to strengthen the market sentiment analysis, a sentiment analysis model or event driven prediction model will be added.

# Future improvement:
1. Moving average will be added
2. Uses more fundamental data to predict the price of stock.
3. Sentiment analysis
4. Train the model with 3000 US stocks.
5. Deep Q learning for portfolio optimization and risk
6. Regularization will be added to avoid overfitting.
7. Quantopian Zipline will be used for backtesting
8. LSTM convolutions network 


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
4. Hyperparameters for dropout has been tested

29/04/2017 Seventh update
1. Hyperparameters for epochs and structure of model have been tested.

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

