# Neural-Network-with-Financial-Time-Series-Data

# Introduction:

Time series Data forms the most paramount part of the quantitative analysis. Today, with the huge amount of data avilable online as well 
as technological advancement, we can analyze large scale data with neural network, which is not available before. Most importantly, this neural network predicts the future movement of the index and acheives a reasonably well result. In this example, we will predict the SP500.

# Content:

This script can download the data of 7 indexes from online provider, form a pandas DataFrame that is compatitble with the TensorFlow library and finally apply a neural network to it. The codes are indifferent to the original code in google cloud platform because of a) update to avoid using depricated codes and b) simplify it for understanding.

# Result:

By running 10000 epochs, with 5 neruons in the first hidden layer, 3 neruons in the second hidden layer and 2outputs, it achieves a 0.737 accuracy.
By running 10000 epochs, with 50 neruons in the first hidden layer, 30 neruons in the second hidden layer and 2 outputs, it achieves a 0.806 accuracy.
By running 10000 epochs, with 50 neruons in the first hidden layer, 30 neruons in the second hidden layer and 2 outputs with dropout rate of 0.2, it achieves a 0.77 accuracy.
By running 50000 epochs, with 50 neruons in the first hidden layer, 30 neruons in the second hidden layer and 2 outputs with dropout rate of 0.2, it achieves a 0.815 accuracy.

# Acknowledgement
Thanks to google, I have created this neural network with tensorflow that can run on any computer without using the google cloud database. The original tutorial is on here https://www.youtube.com/watch?time_continue=1&v=iBs59GlXhIA.

# I WELCOME anyone to work together on this interesting project.

