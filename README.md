# Flight_Delay_Predictions
DecisionTreeClassifier for Kaggle flight delay dataset.

# Objective

Too many times I have seen developers use the Scikit-Learn LabelEncoder() method to encode categorical features in order to feed a DecisionTreeClassifier. As per sklearn documentation, such method should only be used for the TARGET of the model, not for input features. In order to process categorical data properly, sure there is the OneHotEncoder() that can do the job but it gets computationally expensive with larger amounts of possible values and it generates sparse feature matrices.

This project is about predicting if a flight will be delayed by over 15 minutes upon arrival, with Scikit-Learn DecisionTreeClassifier, using US flight data from January 2019 and January 2020. I tried to approach the categorical features problem in a functionnal and effective way. Here is how.

# Environment and tools

The dataset description and dowload can be found here:
https://www.kaggle.com/divyansh22/flight-delay-prediction

This project was run on a jupyter notebook using Pandas, Numpy, Sklearn preprocessing & metrics along with a DecisionTreeClassifier model.

# Project steps

	1. Inspect and visualize the data

In this section, we load the input data into dataframes to gain knowledge of it: dimensions and sizes, correlations, missing values, etc.

	2. Prepare and transform the data

Put the data in a format a machine can learn from by combining disjointed data files into one, removing null values, doing some feature engineering.

	3. Train and predict

After transforming the data, we can start the training process by calling fit() on the sklearn DecisionTreeClassifier model and make predictions on the test set.

	4. Model evaluation and performance

The challenge here was that our data was significantly imbalanced, as flights are way more often on time than delayed. Therefore we needed to build a model capable of effectively separating classes 'on time' or 'delayed'. When data is highly skewed, any model can reach good accuracy by always predicting the same class for example. In our particular case, we are trying to predict the minority class well for the model to be useful. 

Therefore the most relevant metric is AUC (Area Under the Curve):

    If AUC=50% the model is useless as it is wrong 50% of the time.
    If AUC=100% the model is perfect, it identifies both classes right every time.

In conclusion, we get an AUC of 83% on the testing data, meaning that our model performs well at separating classes on unseen data and can predict flight delays effectively (75% accuracy and recall on both classes).
