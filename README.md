# Spotify Time Series Forecasting (ARIMA/SARIMA & LSTM)
**by Aditya Mor**
---

### Project Overview
This project analyzes the key features that influence song popularity in the music industry, using historical data from Spotify, spanning from 1921 to 2020. The dataset includes approximately 170,000 songs with various attributes, ranging from quantifiable features like acousticness, duration, and loudness, to characteristics such as energy, valence, and a popularity score. The goal of this project is twofold: first, to identify the top 4 features that most strongly influence a song’s popularity, and second, to forecast their trends over the next 5-10 years. These insights will help music producers, labels, and streaming platforms make informed decisions about emerging music trends.

To achieve this, I began by cleaning the dataset and enriching it with genre information, which I obtained using Spotify’s API to fetch genre details based on artist names. After performing exploratory data analysis (EDA), a Random Forest model, boosted with XGBoost, was used to identify the top 4 features that best classify a song’s popularity. These features were then subjected to time series forecasting to predict how they will evolve, providing industry stakeholders with insights into potential future trends.

The project utilizes both statistical models (ARIMA/SARIMA) and deep learning models (LSTM) for time series forecasting. By comparing the predictions from these models, the project aims to offer stakeholders reliable insights into the future trajectories of critical music features, such as genre prevalence, explicit content, and acousticness, helping to anticipate shifts in consumer preferences.

This analysis can guide decisions on content creation, curation, and marketing strategies by identifying which features are likely to grow in popularity in the coming years.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset and Spotify API Genre Fetch](#dataset-data-fetch-spotify-api-genre-fetch)
- [Exploratory Data Analysis and Model Preprocessing](#exploratory-data-analysis-and-model-preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#licensing)
  
---
### Installation

To get started with this project, follow these steps:
1. **Clone the repository:**
   '''bash
   git clone git@github.com:enter_your_username/spotify-time-series-forecasting.git
   cd spotify-time-series-forecasting
2. **Download Anaconda Entirely or Just Download Jupyter Notebook:**
   Follow this link to download Anaconda distribution, which includes Jupyter Notebook:
   https://www.anaconda.com/download/success
3. **Launch Jupyter Notebook:**
    Open the notebooks for data-fetching, data-analysis, and models.
4. **Ensure API credentials:**
   If you wish to use the Spotify-API-code.ipynb file to fetch the API genre information yourself, be sure to create your own spotify developer account to have Spotify's API access keys. If you wish to just use my keys, that is alright as well, but I think it's cooler to go through the experience of setting up your own developer account!

Now, you're all set to explore the project!

---
### Dataset, Data fetch, Spotify API Genre Fetch

**Dataset Dictionary:**
1. acousticness: a confidence measure from 0 to 1 how acoustic a song is, with 1 indicating that a song is highly acoustic.
2. artists: the name(s) of the artist(s) who wrote and/or sang the song.
3. danceability: this feature describes how suitable a track is for dancing based on a combination of musical elements. A score of 0 is least danceable and a score of 1 is most danceable.
4. duration_ms: the duration of the song/track in milliseconds.
5. energy: a confidence measure from 0 to 1, which represents how intense and energetic a song is, or how fast, loud, and noisy it is: 0 indicates lowest energy, and 1 indicates highest energy.
6. explicit: a binary feature that describes whether a song is labeled as explicit or not: 1 = yes, 0 = no.
7. id: the song/track's unique identifier.
8. instrumentalness: a measure that describes the instrumentalness of a song. Values closer to 0 indicate songs have vocals, and values near 1 indicate songs have no vocals, or that they are effectively instrumental in nature.
9. key: estimated overall key of the track. Integers map to pitches using standard Pitch Class notation: 0 = C, 1 = C♯/D♭, 2 = D, etc. The value is -1 is no key was detected in the song.
10. liveness: detects a presence of an audience in the recording of the song/track.
11. loudness: metric of how loud a song/track is.
12. mode: indicates the modality (major or minor) of a track, the type of scale from which its  melodic content is derived. Major is represented by 1 and minor is 0.
13. 12. name: name of the song/track.
14. popularity: a score that tracks how popular a song is, from 0 to 100.
15. release_date: the date that the song/track was released.
16. speechiness: metric that tracks the presence of spoken words in a song/track; podcasts, interviews, and such have higher values (closer to 1), and values in the middle (up, down, and around 0.5) describe tracks that contain songs and speech, and values close to or around 0 describe purely music.
17. tempo: measures the beats per minute of a track/song.
18. valence: measures the positivity described in a song/track, from  0 to 1.
19. year: the year that the song/track was released.


**Data Fetch:**
1. In this file, the data was loaded, and some initial EDA was done to check for nulls, the range of the year feature for modularity of the time series implementation later, and some summary statistics. The cleaned file was saved as a csv for loading into the API-genre-fetch file.


**Spotify API Genre Fetch:**
1. There are two files in the data-fetching folder that relate to the API-genre fetch: 1 file that contains the code for your viewing/reusing, and 1 file that contains the output. The reason for two files is because when the code for fetching the genre information is run, the file becomes too large to view on github.
2. If you intend on using this model for any purpose, be sure to download the spotify-api-run.ipynb file. Alternatively, if you just wish to see how I fetched the genre information using Spotify's API, you can view it in the spotify-api-code.ipynb file.

 **How the Genre Information was Fetched:**
1. Importing relevant libraries: pandas, numpy, matplotlib, seaborn, datetime, ConfigManager, logging, requests, base64, concurrent.features, limits, and sleep_and_retry were used.
![api1](https://github.com/user-attachments/assets/a2d3004c-1f6e-49bd-bda3-f04a32376d7c)

2. Load the dataset.
3. Use ConfigManager() to increase the IOPub data rate limit to prevent the jupyter notebook from stopping output when it processes large volumes of data (170,000 rows are present in the dataset, and the API code has to fetch genres for each row, which might have mulitple artist names, so that can be a lot of requests being sent and lots of processing!) or handling frequent API rate limiting messages.
4. Set up logging, created a dictionary (or cache) to store artist-genre mappings, and defined a function to get the API access token from Spotify.
![api2](https://github.com/user-attachments/assets/ab998d26-8bff-45da-b013-1926ff705871)
5. Defined a rate-limiting function that uses that API calls per second, and uses sleep_and_retry to retry the API request if rate limits are reached.
![api3](https://github.com/user-attachments/assets/98cdf44b-83d9-4572-824f-e9572d0ab111)
6. Defined two functions: one that fetches genre for an artist with caching to avoid redudant API calls, and one that ensures that the fetching for genres for a list of artists are done in parallel using caching and rate-limiting.
7. Run the code!
8. Ensure that empty lists in the genre feature are replaced by 'unknown'.
9. Save the file for EDA!

--- 
### Exploratory Data Analysis and Model Preprocessing

**Exploratory Data Analysis:**
1. Initial EDA was performed to check for nulls. Then, a correlational matrix was made to look at which features are highly correlated with each other.
2. Then, a mapping function that categorizes genres into broader genres (like indie-pop, alternative-pop being categorized as just pop) was defined.
3. Since multiple genres were fetched for each song, the mode of a list of genres was chosen as the genre for that song, and if only genre was present, that one was chosen as the genre for that song.
4. Some None values were still present in the genre column and were replaced by 'unknown'.
5. Then a brief statistical analysis of unknown vs known genres was conducted to determine if dropping of unknown genres was appropriate, which it is was not, so the final cleaned dataframe was stored for a Random Forests (boosted with XGBoost) model.

**Model Preprocessing:**
1. In this file, the saved dataframe from the EDA file was imported.
2. Then genre feature was label encoded for classification of the popularity feature. The goal was to determine which top features most correctly classify popularity.
3. A distribution of popularity scores was made to determine the appropriate bin size for the classification model.
4. Most songs had a popularity of scores ranging from 0-40, so the bin sizes were made based on that: 0-35 as 'low', 36-100 as 'high'.
![rf_preprocessing1](https://github.com/user-attachments/assets/c6279bfc-833f-4c78-b7ea-6bc2c78d0e89)
5. Then a distribution of the binned popularity scores was made.
![re_preprocessing2](https://github.com/user-attachments/assets/5de9e687-78f1-4239-a334-b828c8979151)
6. Additional EDA was preformed: unnecessary columns were dropped.
7. Train, test, split was done and X and y were defined. Then X_train and X_test were scaled, disregarding binary columns like explicit and mode and encoded columns such as Labelenc_genre.
8. X and y dataframes were then saved as csv files for importing into the models folder, which contains the rf-model-xgboost.ipynb file.

---
### Modeling

**Random Forests Classifer Boosted with XGBoost:**
1. In this file, first the relevant libraries were imported.
2. Then, the preprocessed data from imported from the preprocessing file in the data-analysis folder. SMOTE was initialized and applied to both X_train and y_train, and their respective distributions were checked to ensure equal proportions. The random forests classifier was then initialized, and an initial cross-validation model was run to check the CV score, which came out to be ~80%.
3. Appropriate regularization parameters were employed to the rf_clf and the model was then fit on the SMOTE-resampled training data.
4. Results after implementation of XGBoost raised test accuracy to 80.28%, but a more in depth analysis of the results of rf classification can be found in the results section of this read me document.
5. Top 6 features were extracted, and of the top 6, the top 4 were selected for time series forecasting.
![image](https://github.com/user-attachments/assets/b2aa9247-29db-402c-a86a-22f4690bb4a6)

**Time Series Forecasting of Top 4 Features: ARIMA/SARIMA**
1. All relevant libraries were first imported.
2. Helper functions were defined as follows: fit_model() (splits data into train, test and prints summary of SARIMA model), train_test() (data before 2000 was train, after 2000 was test), train_RMSE() (calculates the RMSE for the last 40 observations of the training set by comparing the actual values to the model predictions), test_RMSE() (calculates the RMSE for the model on the test data by iterating through the test set and making out-of-sample forecasts and compares the predictions to the actual values), and forecast_model() (fit a SARIMA model on the entire dataset and generates a forecast for the future and calculates the confidence intervals for the forecasted values).
3. Unnecessary features were removed and a dataframe of the annaul averages of the top 4 features was generated to smoothen the data.
4. Line graphs were plotted with year as the modularity, but normalization of the features was required -- this was achieved using MinMaxScaler() and then the features' line plots were plotted.
![image](https://github.com/user-attachments/assets/dfb1b25d-b920-483f-b632-359273892e68)
6. Then, individual dataframes were created and ADF test was performed to determine if the time series is stationary or not; it was concluded that the time sereis was non-stationary.
7. Then, the following steps were performed to create the SARIMA model for each top feature:
   - Use the auto-arima funciton to find the best non-seasonal and seasonal parameters to fit the model for each feature
   - Use the results of the auto-arima function to fit the SARIMA model and get results
   - Compute the train and test RMSE values using the aforementioned helper functions
   - Forecast the model with a 95% confidence interval for the next 10 years, and calculate the expected increase in prevalence for the 1st, 3rd, 5th, 7th, and 9th years to understand the relative increase across the decade.
9. Detailed results of the model are discussed in the results section of this read me file.

**Time Series Forecasting of Top 4 Features: LSTM**
The following steps were conducted for LSTM forecast of the top 4 features:
   - First, individual dataframes were created for each feature with the year feature set as the index.
   - Line plots were generated. (Example of Acousticness is shown below)
     ![image](https://github.com/user-attachments/assets/6b07e165-873d-412d-a4fa-2eebf234025e)
   - Features were normalized using MinMaxScaler() and a function was defined to build sequences from the normalized data for LSTM training.
   - Time step was defined as 10 for acousticness and explicit, and 5 for loudness and labelenc_genre.
   - Data was split into train and test sets.
   - Model layers were added with a dense layer with linear activation for regression output and then the model was compiled and optimized with Adam and the loss function was mean_squared_error.
   - The model was trained on 100 epochs for acousticness, and 40 epochs for explicit, loudness, and labelenc_genre.
   - A plot of the train and validation loss was generated.
   - A plot of true vs predicited values was generated for both train and test sets.
   - A plot of historical, true values, and prediction values was generation for seeing how accurate the forecast was.



     







