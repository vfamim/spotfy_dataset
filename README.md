# Spotify: Predicting Popularity of Songs

![spotify](https://www.scdn.co/i/_global/open-graph-default.png)

## Table of Contents
- [Introduction](#introduction)
- [1. The Business Challenge](#1-the-business-challenge)
- [2. The Dataset](#2-the-dataset)
- [3. Feature Engineering and Variables Filtering](#3-feature-engineering-and-variables-filtering)
- [4. EDA Summary and Insights](#4-eda-summary-and-insights)
- [5. Data Preparation and Feature Selection](#5-data-preparation-and-feature-selection)
- [6. Machine Learning Modelling and Fine Tuning](#6-machine-learning-modelling-and-fine-tuning)
- [7. Business Performance and Results](#7-business-performance-and-results)
- [8. Next Steps](#8-next-steps)
- [9. Lessons Learned](#9-lessons-learned)
- [10. Conclusion](#10-conclusion)
- [References](#references)

<br>

## 1.0. Introduction

This repository contains the solution for a predict challenge: What characteristic of a song that make it popular? 

This project is part of the Kaggle community 

### 1.1. What is Spotify?

Sportify is a digital music, podcast and (recently) videos streaming that gives access to millions of songs and other content from artists all over the world.

### 1.2. Spotify Business Model

Spotify operates under a freemium business model (basic services are free, while additional features are offered via paid subscriptions). Spotify generates revenues by selling premium streaming subscriptions to users and advertising placements to third parties.

### 1.3. Project Development

The project was developed based on CRISP-DS (Cross-Industry Standard Process - Data Science) project management method.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/1024px-CRISP-DM_Process_Diagram.png" alt="CRISP" style="zoom:30%;" />

### 1.4. The Goal of Project

Create a model who predict a Spotify songs popularity based on songs characteristics.

### 1.5. Solution

In this project, a regression model will be used to predict the popularity index of a Spotify songs.

## 2.0. The Dataset

### 2.1. Dataset Source

Dataset is available on Kaggle community: [Spotify Dataset 1921-2020, 160k+ Tracks | Kaggle](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks)

### 2.2. Dataset Shape

The dataset has 170653 rows and 19 columns.

### 2.3. Columns Descriptions

*Numerical:*

  - **Acousticness** (Ranges from 0 to 1): The relative metric of the track being acoustic. 1.0 represents high confidence the track is acoustic
  - **Danceability** (Ranges from 0 to 1): The relative measurement of the track being danceable. Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
  - **Energy** (Ranges from 0 to 1): The energy of the track. Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy
  - **duration_ms** (Integer typically ranging from 200k to 300k): The length of the track in milliseconds (ms)
  - **instrumentalness** (Ranges from 0 to 1): The relative ratio of the track being instrumental. Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0
  - **valence** (Ranges from 0 to 1): A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
  - **popularity** (Ranges from 0 to 100)
  - **tempo** (Float typically ranging from 50 to 150): The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration
  - **liveness** (Ranges from 0 to 1): Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live
  - **loudness** (Float typically ranging from -60 to 0): The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db
  - **speechiness** (Ranges from 0 to 1): Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
  - year (Ranges from 1921 to 2020)

*Dummy:*

  - **mode** (0 = Minor, 1 = Major): Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0
  - **explicit** (0 = No explicit content, 1 = Explicit content): Whether or not the track has explicit lyrics ( true = yes it does; false = no it does not OR unknown)

*Categorical:*

  - **key** (All keys on octave encoded as values ranging from 0 to 11, starting on C as 0, C# as 1 and so on…)
  - **artists** (List of artists mentioned)
  - **release_date** (Date of release mostly in yyyy-mm-dd format, however precision of date may vary)
  - **name** (Name of the song)

### 2.4. Exploratory Data Analysis

<puting data_analisys here>

## 3.0. Feature Engineering

### 3.1. Hypothesis map

The map below help us to decide which variables we need in order to validate the hypotheses.

<putting hypotesis map>

### 3.2. Hypothesis Creation

| #    | Audio Feature Object                          |
| ---- | --------------------------------------------- |
| 1.   | Popularity occur with high `acousticness`.    |
| 2.   | Popularity occur with high `danceability`.    |
| 3.   | Popularity occur with 80% of `liveness`.      |
| 4.   | Popularity occur with `loudness` above -10.   |
| 5.   | Popularity occur with low `energy`.           |
| 6.   | Popularity occur with low `speechiness`.      |
| 7.   | Popularity occur with high `valence`.         |
| 8.   | Popularity occur with `key` equal 2.          |
| 9.   | Popularity occur with `mode` equal 1.         |
| 10.  | Popularity occur with high `tempo`.           |
| 11.  | Popularity occur with low `instrumentalness`. |

| #    | Track                                     |
| ---- | ----------------------------------------- |
| 1.   | Popularity occur with `explicit` equal 1. |

| #    | TIME                                       |
| ---- | ------------------------------------------ |
| 1.   | Popularity occur with high `duration_min`. |

## 4.0. Univariate Analysis

### 4.1. Target Variable

<img01>

The songs popularity are concentrated at zero, most of the songs do not have good popularity score.

The target variable has a multimodal distribution.

<img02>

Songs with 0 score are from 1920s or maybe older, it has anything to do with the popularity measurement system, this means all songs related to an artist are non popular.

### 4.2. Numerical Variables Distribution

<img03>

Some observations:

* `loudness` has negative values and left skew
* `liveness`, `speechiness` and `duration_min` is right skew
* `loudness` has negative skew
* `tempo` has a unimodal distribution
* `acousticness` is a bimodal dataset

### 4.3. Bivariate Analysis

The bivariate analysis consists of the independent variable analysis with respect to the target variable. 

#### H1. Popularity occur with high `acousticness`.
* FALSE
*  HIGH RELEVANCE
<img05>
#### H2. Popularity occur with high `danceability`.
* TRUE
* HIGH RELEVANCE
<img06>
#### H3. Popularity occur with 80% of `liveness`.
* FALSE
* LOW RELEVANCE
* <img07>
#### H4. Popularity occour with `loudness` above -10.
* TRUE
* HIGH RELEVANCE
<img08>
#### H5. Popularity occur with low `energy`.
* FALSE
* HIGH RELEVANCE
*<img09>
#### H6. Popularity occur with low `speechiness`.
* TRUE
* LOW RELEVANCE
<img10>
#### H7. Popularity occur with high `valence`.
* FALSE
* HIGH RELEVANCE
<img11>
#### H8. Popularity occour with `key` equal 2.
* DEPEND
* LOW RELEVANCE
<img12>
#### H9. Popularity occur with `mode` equal 1.
* FALSE
* HIGH RELEVANCE
<img13>
#### H10. Popularity occur with high `tempo`.
* TRUE
* LOW RELEVANCE
<img14>
#### H11. Popularity occur with low `instrumentalness`.
* TRUE
* LOW RELEVANCE
<img15>
#### H12. Popularity occur with `explicit` equal 1.
* TRUE
* LOW RELEVANCE
<img16>
### H13. Popularity occur with high `duration_min`.
* FALSE
* LOW RELEVANCE
<img17>

### 4.4. Multivariate Analysis

The main goal of the multivariate analysis is to check how variables are related.

<img18>

## 5.0. Data Preparation

There are mainly three types of data preparation:

1. **Normalization**: A scaling technique which values are shifted and rescaled so that they end up ranging between 0 and 1.
2. **Standardization**: Scaling technique where the values are centered around the mean with a unit standard deviation.

### 5.1. Standardization

None of the numerical variables have a normal distribution, therefore the Standardization technique will not be applied.

### 5.2. Rescaling

<img04>

The boxplot above shows the variables with a high influence of outliers. Two rescaling techniques will be applied: the Min-Max Scaler and the Robust Scaler.

The Min-Max Scaler is applied in non-Gaussian distributions. However, it is susceptible to outliers, that is, if the feature has a high outlier influence, than the Min-Max Scaler will tend to result in distorted numbers due to the outliers. That happens because it uses the maximal and minimal values (range) to rescale the numbers.

The Robust Scaler is also applied to non-Gaussian distributions, and performs better for variables with outliers, because it scales the data with the range of the first quartile (25th quantile) and the third quartile (75th quantile) of the IQR (Interquartile Range).

## 6.0. Feature Selection

The first step of feature selection was to select the features for the machine learning training: 

```
['valence', 'year', 'acousticness', 'danceability','energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity','speechiness', 'tempo', 'duration_min']
```

The second step,  drop the target variable ['popularity'] in order to allow the model to be trained.

The third step was to apply Boruta to determine the most relevant features:
```
['year',  'acousticness',  'danceability',  'energy',  'instrumentalness',  'liveness',  'loudness', 'speechiness', 'tempo', 'duration_min']
```

## 7.0 Machine Learning Modelling 

In order to solve the task 3 regression models will be used:

* Linear Regression
* Random Forest Regression
* XGBoost Regressor

### 7.1. Single Performance

<single performance>

The metric chosen is RSME (root mean square error) and Random Forest Regression has the best results.

### 7.2 Real Performance

The technique to validating our models is called Cross-Validation, it is a resampling procedure used to evaluate machine learning models.  The goal of cross-validation is to test the model's ability to predict new data that was not used in estimating in, in order to flag problems like overfitting. Here are the results:

<real_performance>

XGBoost Regressor has the best RSME, so based on the business context and in order to better accomplish the project goals, the chosen model to perform the fine tuning is XGBoost.

## 8.0. Fine Tuning

The hyperparameter fine-tuning is performed in order to improve the model performance in comparison to the model with default hyperparameters.

There are two ways to perform hyperparameter fine-tuning: through grid search or through random search. In grid search all predefined hyperparameters are combined and evaluated through cross-validation. It is the best way to find the best hyperparameters combinations, however it takes a long time to be completed. In random search, the predefined hyperparameters are randomly combined and then evaluated through cross-validation. It may not find the best optimal combination, however it is much faster than the grid search and it is largely applied.

In this project, the chosen technique is GridesearchCV, the results are:

```
{'colsample_bytree': 0.9,
 'learning_rate': 0.03,
 'max_depth': 5,
 'min_child_weight': 4,
 'n_estimators': 1500,
 'objective': 'reg:squarederror',
 'silent': 1,
 'subsample': 0.7}
```

### 8.1. Final Model

<final_model>

<final_model_cv>

It was observed that was a significant decrease in RSME.

## 9.0. Machine Learning Performance

<img19>

Some comments related to the machine learning performance.

### 9.1 Popularity x Prediction

<img20>

the graphics shows that the prediction and `popularity` have a very close line, which means the prediction have the same shape of `popularity` line. The shadows represents a variance of several predictions.

### 9.2. Error Rate

<img21>

The error rate graphics shows the error rate of the predictions at each time period. There is a large error rate in the early years, this due to the amount of old songs with very low popularity.

<img24>

Checking the subsequent years it is notice a few predictions below one, this values represent and underestimated prediction, the values above one represents overestimated predictions.

### 9.3. Error Distribution

<img22>

The error distribution almost follows a normal distribution.

### 9.4. Scatterplot Error

<img23>

The zero values `popularity` brought some negative error rate. The points seems well fit in a horizontal tube which means that there's a few variation in the error. If the points formed any other shape (e.g opening/closing cone or an arch), this would mean that the errors follows a trend and we would need to review our model.

## 10.0. Next Steps

* Experiment with other Machine Learning algorithms to improve business performance.
* Experiment with selecting other features to see how much the RMSE is impacted.
* Experiment with other hyperparameter fine-tuning strategies to see how much the RMSE is impacted.
* Improve bot or application to user interaction.

 

