# Spotify Time Series Forecasting (ARIMA/SARIMA & LSTM)
**by Aditya Mor**
---

## Project Overview
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
- [License](#licensing)
  
---
## Installation

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
## Dataset, Data fetch, Spotify API Genre Fetch

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
## Exploratory Data Analysis and Model Preprocessing

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
## Modeling

### Random Forests Classification

**Random Forests Classifer:**
1. In this file, first the relevant libraries were imported.
2. Then, the preprocessed data from imported from the preprocessing file in the data-analysis folder. SMOTE was initialized and applied to both X_train and y_train, and their respective distributions were checked to ensure equal proportions. The random forests classifier was then initialized, and an initial cross-validation model was run to check the CV score, which came out to be ~80%.
3. Appropriate regularization parameters were employed to the rf_clf and the model was then fit on the SMOTE-resampled training data.
4. Results after implementation of XGBoost raised test accuracy to 80.28%, but a more in depth analysis of the results of rf classification can be found in the results section of this read me document.
5. Top 6 features were extracted, and of the top 6, the top 4 were selected for time series forecasting.
![image](https://github.com/user-attachments/assets/b2aa9247-29db-402c-a86a-22f4690bb4a6)

### Time Series Forecasting

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

---
## Results

### Random Forests Classification

**Random Forests Classification: Initial CV, RF (unboosted), RF (Boosted w/ XGBoost):**

**Initial CV:**
1. Initial Cross Validation Scores were computed for comparing the performance of the random forests model:
![image](https://github.com/user-attachments/assets/2db36cb5-427a-41a1-8609-d3e74b0f5430)

The average CV score ~80%. A tuned RF-XGBoosted model should perform as well or better than this score. 

**RF (unboosted):**

1. After setting the regularization parameters and running the first iteration of the random forests classifier (without XGBoost), the results were as follows:
![image](https://github.com/user-attachments/assets/2753f9df-31bd-45d0-9c2f-852854cb7279)
  - Train Accuracy (SMOTE resampled): 91.92%
  - Test Accuracy: 79.24%
  - F1-score (high bin and low bin, respectively): 78%, 80%
  - Recall (high bin and low bin, respectively): 79%, 80%
2. AUC-ROC Score was 0.88:
  - The ROC curve and its associated AUC (area under the curve) plots the recall against the false positive rate; this is shown below.
    ![image](https://github.com/user-attachments/assets/8545c361-091e-4504-b02b-9d672e82f52d)
  - An AUC-ROC score of 0.88 means that the model has a high ability to distinguish between the high popularity bin and the low popularity bin class. A score closer to 1 indicates a perfect classifier and a score closer to 0.5 indicates a random guess; therefore, a score of 0.88 indicates that the model classifies very well.

These results demonstrate a strong performance of the random forest model in classifying song popularity into two bins (high and low). The high train accuracy indicates that the model fits well to the resampled training data, suggesting effective learning. The application of SMOTE helped to mitigate the class imbalance, especially for the underrepresented bin, ensuring the model doesn't favor the majority class.

However, the slight drop to 79.24% in test accuracy reflects a more realistic estimate of the model’s performance when applied to unseen data. This decrease could indicate slight overfitting to the training set, which is common when working with resampled data. The performance is still strong, but further tuning may reduce this gap.

The F1-scores for both bins (78% for the high bin and 80% for the low bin) are relatively balanced, indicating that the model is capable of performing well on both classes without a significant bias towards one. F1-score is a good measure in this context because it balances precision and recall, which are both important when classifying in the presence of imbalanced data.

The recall values for both bins (79% and 80%) demonstrate that the model is retrieving most of the relevant instances for both popularity categories, which is important when the cost of misclassification is high (e.g., misclassifying a popular song as unpopular).

Overall, these metrics suggest that the model generalizes well and effectively addresses the class imbalance problem, with a robust trade-off between precision and recall across both classes.

**What could these metrics mean when compared with the initial CV score of 80%?**
- The CV Mean score of 80% provides an average estimate of model performance during k-fold cross-validation, meaning the model's ability to generalize has been tested on multiple, unseen validation sets.
- And this aligns well with the test accuracy of 79.24% because it suggests that the model is performing consistently, and it gives confidence that the model is not overfitting too much and generalizes well to unseen data.


**RF (Boosted w/ XGBoost):**

1. Employment of XGBoost and its results metrics are shown below:
![image](https://github.com/user-attachments/assets/f6331af5-0a75-407f-aafb-68f6b4c5a498)
  - Test Accuracy: 80.28%
  - F1-score (high bin and low bin, respectively): 79%, 80%
  - Recall (high bin and low bin, respectively): 80%, 80%
  - AUC-ROC Score: 0.88

Slight Improvement in Test Accuracy: The boosted RF model achieved a slightly better test accuracy (80.28%) compared to the initial Random Forest classifier (79.24%). This suggests that incorporating XGBoost helped fine-tune the model’s performance, resulting in a slight but meaningful improvement in accuracy.

Consistent Recall and F1-Scores: The recall and F1-scores for both bins are very close to the original model’s values.

Balanced Performance: The nearly equal performance across both high and low bins in terms of recall, precision, and F1-scores indicates that the model is well-balanced, with no clear bias towards either class.

Conclusion:
The XGBoosted Random Forest model slightly outperforms the standard Random Forest model and maintains a robust, balanced performance across both popularity bins. The AUC-ROC score of 0.88 reflects the model's excellent ability to distinguish between high and low popularity songs. These results suggest that tuning and incorporating boosting with XGBoost adds value to the model’s overall classification ability, providing a more powerful predictor for song popularity.

### Time Series Forecasting

**Time Series Forecasting of Top 4 Features: ARIMA/SARIMA:**
1. In this file, the features were first visualized without normalization and then with normalization to see their movements through time before breaking each feature down into a time series.
![image](https://github.com/user-attachments/assets/ab3033fc-e2d6-4aaa-a6d2-be4479f8e02b)
2. Shown below are the time series results of each individual feature and its respective analysis:

### Acousticness Prevalence:
- Auto-arima parameters:
![image](https://github.com/user-attachments/assets/fd5bce7d-9ee7-4217-b446-e5b6124f4a13)
- Fitting the best model parameters into SARIMA:
![image](https://github.com/user-attachments/assets/60257ea1-3b47-4e2c-b590-8746eb672ee7)
- Train RMSE:
  Value: 0.05358
![image](https://github.com/user-attachments/assets/81cc4355-2af0-48e6-ae9c-264f5c3ddd35)
- Test RMSE:
  Value: 0.02066
![image](https://github.com/user-attachments/assets/739355cc-6472-4145-a590-7c269406857e)
- Total Expected Increase in Prevalence:
![image](https://github.com/user-attachments/assets/8e3767a6-9fc9-4476-b535-19766701ef7e)

### Explicit Prevalence:
- Auto-arima parameters:
![image](https://github.com/user-attachments/assets/9cb2fd21-fa5d-401e-b31f-b7d220eceaf5)
- Fitting the best model parameters into SARIMA:
![image](https://github.com/user-attachments/assets/18b064ba-11a2-46ff-93d5-0f51b6dc69a6)
- Train RMSE:
  Value: 0.05890
![image](https://github.com/user-attachments/assets/d7375a61-d2da-4ec7-ba02-1c89d014171e)
- Test RMSE:
  Value: 0.13249
![image](https://github.com/user-attachments/assets/6cae5ede-60ee-4942-a0f2-2f1a42335708)
- Total Expected Increase in Prevalence:
![image](https://github.com/user-attachments/assets/6cf3d655-7628-4ae8-a254-7f622f9b12c0)

### Loudness Prevalence:
- Auto-arima parameters:
![image](https://github.com/user-attachments/assets/c6e5c6c1-322a-4566-9166-f9c60998ddc5)
- Fitting the best model parameters into SARIMA:
![image](https://github.com/user-attachments/assets/ba1b3860-a119-487d-9a79-ed2058053bfa)
- Train RMSE:
  Value: 0.04879
![image](https://github.com/user-attachments/assets/00ae9a20-b361-47f7-94a8-f1cefa255814)
- Test RMSE:
  Value: 0.04779
![image](https://github.com/user-attachments/assets/f4d79f94-0112-4f32-8a8f-02625a5e2371)
- Total Expected Increase in Prevalence:
![image](https://github.com/user-attachments/assets/39d397d6-edc5-4342-80f8-38bbe3cc0d8c)

### Label Encoded Genre Prevalence:
- Auto-arima parameters:
![image](https://github.com/user-attachments/assets/511ff2ff-0f62-415a-9cd9-aa84fbfb8b54)
- Fitting the best model parameters into SARIMA:
![image](https://github.com/user-attachments/assets/b8daa6d9-cf62-4057-a595-a271e1713582)
- Train RMSE:
  Value: 0.07149
![image](https://github.com/user-attachments/assets/1434f65a-4a15-4573-be77-46fd0a632a08)
- Test RMSE:
  Value: 0.03821
![image](https://github.com/user-attachments/assets/656726a1-1895-4af3-bb72-b2ac8a44187b)
- Total Expected Increase in Prevalence:
![image](https://github.com/user-attachments/assets/a6bd5d81-63ae-4609-b5df-5fc75647679c)

### Top 4 Features Increase in Prevalence:
![image](https://github.com/user-attachments/assets/f7c5bb25-8454-4479-aa25-5f33e082500c)

### Based on our ARIMA/SARIMA time series forecasting model, the projected changes over the next 9 years for various features are as follows:

Explicit: 641.30% Increase over 9 Years
Loudness: 558.75% Increase over 9 Years
Labelenc_genre: 50.79% Increase over 9 Years
Acousticness: 2.023% Increase over 9 Years

**Strategic Insights for Future Music Trends**
The analysis of future trends reveals significant shifts that could be crucial for the music industry.

Explicit Content (641.30% Increase):
This explosive increase in explicit content implies that listeners are likely to prefer music with more direct and unfiltered expression. This aligns with the ongoing trend of more bold and authentic lyrical content in popular music.

Strategic Recommendation:
Music executives should prioritize artists and genres that embrace explicitness, as this will likely resonate more with future audiences. This could also signal a shift in cultural attitudes towards openness in media consumption.

Loudness (558.75% Increase):
Loudness reflects the intensity and energy of the music. The sharp increase suggests that future music will favor higher-energy tracks, possibly driven by younger audiences and the dominance of streaming platforms that favor engaging, high-volume tracks.

Strategic Recommendation:
Labels should focus on high-energy tracks and artists that emphasize loud, dynamic production. This could be crucial for success in environments where energetic music dominates playlists and social media platforms.

Labelenc_genre (50.79% Increase):
While genre diversity is projected to grow more moderately than explicit content or loudness, it remains an important factor. The increase of 50.79% implies that genre diversity will still play a significant role in the industry, although it won't grow as rapidly.

Strategic Recommendation:
Continue to support artists who experiment with genres but with a clear understanding that genre-blending and crossover styles may not drive the industry's future as much as energy and explicitness.

Acousticness (2.023% Increase):
Acoustic music is expected to see only a modest growth of 2.023%, indicating that more natural and stripped-back music styles will not be the leading trend. However, this could be an opportunity for artists and labels to differentiate themselves in a highly competitive market dominated by louder and more explicit content.

Strategic Recommendation:
For artists or labels looking to stand out, investing in acoustic music could be a niche yet impactful opportunity, providing an avenue to capture audiences seeking authenticity and emotional depth in an otherwise high-energy landscape.

Visual Comparison
The following graphs compare the Actual Train Data vs. Predicted Prevalence for key features:

Acousticness: Shows a clear decline in prevalence over time, with the model predicting a steady but very modest increase in the future, matching the 2.023% growth forecast.

Explicit Content: Displays a clear trend of exponential growth, consistent with the model's forecast of a 641.30% increase. The model fits this trend well and predicts further upward growth in explicit tracks.

Loudness: The model predicts a steady rise in loudness, with a near-perfect fit between actual and predicted values, supporting the forecasted 558.75% increase.

Labelenc_genre: Genre diversity has fluctuated over time, but the model predicts steady growth moving forward, aligning with the 50.79% increase forecast.

RMSE Performance
To measure model performance, RMSE (Root Mean Squared Error) was calculated for both the train and test data across all features:

Acousticness: Train RMSE: 0.05358, Test RMSE: 0.02066
Explicit Content: Train RMSE: 0.05890, Test RMSE: 0.13249
Loudness: Train RMSE: 0.04879, Test RMSE: 0.04779
Labelenc_genre: Train RMSE: 0.07149, Test RMSE: 0.03821
These RMSE values indicate the predictive accuracy of the model. Lower test RMSE values suggest that the model performed well in predicting future trends, particularly for Loudness and Acousticness, which show closely matching train and test RMSE scores.

Conclusion
The time series forecasting analysis reveals that Loudness and Explicit Content are set to dominate future music trends, with substantial growth projected over the next 9 years. For music executives and labels, these insights highlight the importance of prioritizing high-energy, explicit tracks in order to stay ahead of consumer demand.

At the same time, genre diversity remains relevant, although it will grow at a slower pace. Acoustic music, while not a major trend, presents a unique opportunity for differentiation in a crowded market.

This analysis showcases the power of predictive modeling in the music industry, providing actionable insights for strategic planning, artist selection, and long-term content creation.

**Time Series Forecasting of Top 4 Features: LSTM:**
1. True values vs Predicted Values for each of the top features is shown below:

**Acousticness:**

![image](https://github.com/user-attachments/assets/1823dea2-80dc-4674-bd78-57a706f644c2)

![image](https://github.com/user-attachments/assets/1c772717-60ad-43b0-85e7-9bebebf3543b)

![image](https://github.com/user-attachments/assets/1f86d5c3-3b17-4ddc-84b2-a157cd8bfbb7)

**Explicit:**

![image](https://github.com/user-attachments/assets/aff6e585-0324-491e-b35a-1b1e639fe1b0)

![image](https://github.com/user-attachments/assets/fe01d8f3-07eb-4248-9f84-a2ab525fab32)

![image](https://github.com/user-attachments/assets/eefe613d-2bc5-472a-bc05-77cbbbfbc75f)


**Loudness:**

![image](https://github.com/user-attachments/assets/6789867f-b10b-4f3a-b8f4-52fd6d83cbb3)

![image](https://github.com/user-attachments/assets/e12b38a9-3e76-4e1d-a3c0-64c49cfcff1f)

![image](https://github.com/user-attachments/assets/2476ed9e-0343-41a4-8879-04a5e397e22a)

**Genres:**

![image](https://github.com/user-attachments/assets/14d0c437-aaa1-426d-b30a-a660d581fe2d)

![image](https://github.com/user-attachments/assets/6f53fe0e-2c83-4601-918e-99f3f3cbb751)

![image](https://github.com/user-attachments/assets/c94a9a71-b163-4ccc-b779-8b4bd4f3b0a0)

## Comparison of ARIMA/SARIMA vs. LSTM:

### Acousticness:

LSTM Model:

The LSTM model displayed decent performance on the train set but struggled with the test set, where the model flattened out its predictions. This shows that LSTM may have had difficulty capturing the sharp fluctuations in acousticness over time.
The predictions suggest that acousticness may remain fairly static in the future.

ARIMA/SARIMA:

The ARIMA model, which had a forecasted growth of 2.023% over nine years, provided better alignment with actual historical data but also forecasted a minimal increase in acousticness, supporting the idea that acoustic music will not see a major resurgence in the future.

Conclusion:

Both models indicate a minimal growth trend for acousticness, with LSTM showing difficulty in capturing complex patterns and ARIMA/SARIMA showing a modest but steady prediction. This aligns with the industry shift away from acoustic music, making it a niche area moving forward.

### Explicit 

LSTM Model:

The LSTM model showed strong performance in predicting explicit content on the train set, capturing the sharp rise in the number of explicit songs over time. However, on the test set, the model tended to smooth the predicted curve, underestimating the sharp peaks in explicit song counts.

ARIMA/SARIMA:

The ARIMA model predicted a substantial increase of 641.30% over the next nine years, closely aligning with historical trends. The ARIMA model captured the exponential growth better than the LSTM model.

Conclusion:

Both models agree on the rapid growth of explicit content in music. ARIMA/SARIMA, however, provided a more accurate reflection of the explosive trend, while LSTM slightly underestimated it. The focus on explicit music is likely to continue dominating the industry as predicted.

### Loudness:

LSTM Model:

The LSTM model performed well on both the train and test sets for loudness, showing strong alignment between the true and predicted values. It was able to capture the overall upward trend in loudness, although it showed some smoothing during test predictions.

ARIMA/SARIMA:

The ARIMA model predicted a significant 558.75% increase in loudness over the next nine years, closely matching the historical upward trajectory of loudness in music.

Conclusion:

Both models performed well in capturing the rise in loudness, with ARIMA/SARIMA showing a slight edge in predicting the more substantial long-term increase. The industry is likely to continue favoring high-energy, louder music moving forward.

### Genres (Label Encoded):

LSTM Model:

The LSTM model performed reasonably well on the train set for genres but, similar to other features, showed some smoothing on the test set, where it underestimated the volatility in genre prevalence. However, it still managed to reflect general trends.

ARIMA/SARIMA:

The ARIMA model predicted a 50.79% increase in genre diversity, capturing the fluctuations and providing a moderate outlook on genre prevalence over time.

Conclusion:

While both models showed the continued importance of genre diversity, the growth will be moderate compared to other features. Genre variety will remain important, but explicit content and loudness will likely outpace it in driving future trends.

## Final Conclusion:

The results of both the ARIMA/SARIMA and LSTM models provide valuable insights into the future of music trends:

Explicit Content is expected to dominate the industry with the largest projected increase, signaling a shift towards more bold, uncensored music.

Loudness will continue to rise, reflecting the demand for high-energy tracks.
Genre Diversity will grow moderately but will not drive the industry as strongly as explicit content or loudness.

Acousticness is forecasted to see minimal growth, remaining a niche area in the future.
Overall, the ARIMA/SARIMA models performed slightly better at capturing long-term trends, while the LSTM models showed promise but tended to smooth predictions for volatile features. The final conclusion is that explicit and loud music will dominate future trends, while acoustic music will see limited growth. Genre diversity will continue to be relevant but not as influential as the other features.

These predictions can help music executives and labels strategize by focusing on explicit, high-energy content while identifying potential niche opportunities in acoustic music. Both models demonstrate the value of predictive modeling in understanding and anticipating changes in the music landscape.

--- 

## Licence
This project is licensed under the MIT License - see the LICENSE file for details.




