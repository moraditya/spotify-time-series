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
- [How to Use](#how-to-use)
- [Contributing](#contributing)
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

**Conclusion:** 

Strategic Insights for Future Music Trends Based on Forecasting Results
Based on our time series analysis and forecasting, it's evident that Loudness and Explicit content are poised for substantial growth in the coming years, with respective increases of 558.75% and 641.30% over the next 9 years. This suggests a significant shift towards more intense, high-energy, and explicit tracks dominating the industry, likely reflecting changing consumer preferences and the evolution of popular music.

For music executives or labels focused on following market trends, this data is pivotal. Prioritizing artists and genres that emphasize loudness and explicit content may provide a competitive advantage, as these attributes are projected to resonate more with future audiences. If you're looking to sign or promote artists that align with rising trends, these two features should be at the forefront of your considerations.

However, Labelenc_genre, which represents genre diversity and classification, shows a moderate increase of 50.79%. This implies that genre variety will still play an important role, although not as sharply as the intensity or explicitness of music. Labels should continue to support genre experimentation but with an understanding that it may not outpace the growth of other, more aggressive features.

On the other hand, Acousticness, which relates to more natural, less electronic music, is projected to see only a modest increase of 2.023% over the next 9 years. While it may appear that acoustic music won't be a major trend, this low growth forecast presents a unique opportunity for labels and artists who want to break away from the mainstream. With the music industry leaning heavily towards louder, more explicit content, investing in acoustic or more stripped-back styles could provide a way to stand out in a crowded market.

In conclusion, the time series model clearly indicates that Loudness and Explicit content will dominate future music trends. For executives looking to ride the wave of popularity, focusing on these features is key. However, for those aiming to disrupt the industry with a fresh perspective, the low but steady rise in acoustic music offers a niche, yet impactful opportunity to create a distinct identity and capture an audience seeking authenticity and emotional depth in their music experience.

This analysis provides an invaluable use case for predictive modeling in the music industry and highlights the balance between following trends and disrupting the market. This level of insight can be directly utilized in strategic planning, artist selection, and long-term content creation.

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




