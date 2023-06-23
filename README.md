# network_traffic_prediction
In this project we propose a conolutional LSTM based model to predict the future network traffic using the past data. We have used the mobile traffic data set released through Telecom Italia’s Big Data Challenge which includes network activity measurements in terms of total cellular traffic volume for the city of Milan and the Trentino region [Barlacchi et al.2015]. The available data is collected from 1 Nov 2013 through 1 Jan 2014 over 10-minute intervals. For our work, we have used the normalized mobile traffic data of the city of Milan. The first 50 days of data of is used for training and validation and the rest of the data is used for testing the performance of the model. The original data set partitions the city of Milan into 100x100 squares of size 235m by 235m. For our study, we have used the data of the cells[60:70][50:60], which contains 100 cells and shapes a 2350mx2350m square area. We try to predict the user density at different time stamps as accurately as possible via different neural network architectures.

Validation and test datasets are provided in separate files. The training file can be find viw the follwoing link:
[Training Data](https://drive.google.com/file/d/1YEOXfZgFGn3AhPFxiC2PfjXW0qMNtjaq/view?usp=share_link)

Each file containg 3 rows of data, including cell ID, time stamp and the network connection requests.
