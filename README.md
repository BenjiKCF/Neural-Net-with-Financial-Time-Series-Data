# Neural-Network-with-Financial-Time-Series-Data
An evnet driven model that uses financial time series data with New York Times information to form a LSTM recurrent neural network.

# Versions
There are 3 models. The first 2 models are based on price and volume data alone. The third model is an event driven model that uses sentiment analysis on news data to predict the stock price. 

1. Prediction with 22 previous days

Filename: LSTM_Stock_prediction_20170507.ipynb

Currently not working, but the model is optimized.

2. Prediction with 22 previous days

Filename: LSTM_Stock_prediction_20170528(Quandl).ipynb

Using Quandl Database instead of pandas datareader. Not optimized.

3. Event driven model with 3 previous days (Using CSV file for price data)

Filename: Event_driven_LSTM_Stock_prediction.ipynb

Using News for sentiment analysis for that day.


# Future improvement:
1. Optimize hyperparameter
2. Technical indicator will be added to the features
3. More data will be trained
4. Quantopian Zipline will be used for backtesting
5. Twitter sentiment analysis will be added
6. More indexes will be included

# Result:
Lastest LSTM model result for 7 years of testing data that has not been trained:

![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/Photos/20170510result.png)

Train Score: 0.00006 MSE (0.01 RMSE)

Test Score: 0.00029 MSE (0.02 RMSE)

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

08/05/2017 Seventh update
1. All Hyperparameters have been tested and results have been uploaded.
2. Fixed comment for the data loader
3. More technical analysis like volume, moving average and other indexes will be added

28/05/2017 Eighth update
1. Using Quandl instead of Pandas datareader
2. Correlation heatmap has been addded
3. Using Adjusted OHLCV for the network
4. All functions can be loaded from lstmstock.py
5. A Quandl api key is provided temporarily for those who do not own a quandl account
6. Moving averages have been added

02/10/2017 Nineth update (Big update)
1. Event driven analysis
2. Switched to Tensorflow LSTM model

# How to use Quandl
With this link, you should be able to get the historic price data of a particular stock after login. 
Use Export > Python > api key and insert the api key to your model.
https://www.quandl.com/product/WIKIP/WIKI/PRICES-Quandl-End-Of-Day-Stocks-Info
![Alt text](https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/Photos/quandl.png)

# References:
Bernal, A., Fok, S., & Pidaparthi, R. (2012). Financial Market Time Series Prediction with Recurrent Neural Networks.

Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.

Jaeger, H. (2001). The “echo state” approach to analysing and training recurrent neural networks-with an erratum note. Bonn, Germany: German National Research Center for Information Technology GMD Technical Report, 148(34), 13.

Jaeger, H. (2002). Tutorial on training recurrent neural networks, covering BPPT, RTRL, EKF and the" echo state network" approach (Vol. 5). GMD-Forschungszentrum Informationstechnik.

Maass, W., Natschläger, T., & Markram, H. (2002). Real-time computing without stable states: A new framework for neural computation based on perturbations. Neural computation, 14(11), 2531-2560.
