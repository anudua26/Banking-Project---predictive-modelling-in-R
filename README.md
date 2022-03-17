[bank-full_train (1).csv](https://github.com/anudua26/Banking-Project---predictive-modelling-in-R/files/8294800/bank-full_train.1.csv)
[bank-full_train.csv](https://github.com/anudua26/Banking-Project---predictive-modelling-in-R/files/8294840/bank-full_train.csv)
[Prediction using model.csv](https://github.com/anudua26/Banking-Project---predictive-modelling-in-R/files/8294867/Prediction.using.model.csv)
# Banking-Project---predictive-modelling-in-R

Business Scenario:

A local bank is rolling out term deposit for its customers. They have in the past connected to their customer base through phone calls. Results for these previous campaigns were recorded and have been provided to the current campaign manager to use the same in making this campaign more effective.

Challenges that the manager faces are following:

•  Customers have recently started to complain that bank’s marketing staff bothers them with irrelevant product calls and this should immediately stop

•  There is no prior framework for her decide and choose which customer to call and which one to leave alone

She has decided to use past data to automate this decision, instead of manually choosing through each and every customer. Previous campaign data which has been made available to her; contains customer characteristics , campaign characteristics, previous campaign information as well as whether customer ended up subscribing to the product as a result of that campaign or not. Using this she plans to develop a statistical model which given this information predicts whether customer in question will subscribe to the product or not. A successful model which is able to do this, will make her campaign efficiently targeted and less bothering to uninterested customers.

We are given you two datasets - bank-full_train.csv and bank-full_test.csv . Need to use data bank-full_train to build predictive model for response variable “y”. bank-full_test data contains all other factors except “y”, you need to predict that using the model that you developed and submit your predicted values in a csv files.

Evaluation Criterion :KS score on test data. larger KS, better Model

The predictions should not contain any NA values. Can use any predictive modelling technique


DESIRED RESULT from this project:  

KS score for test data should come out to be more than 0.47



I have used the Logistic Regression predictive modelling technique for this project. I have attached my R script in this project repository.

