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
<br>

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