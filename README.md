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

---

**Data Fetch:**
1. In this file, the data was loaded, and some initial EDA was done to check for nulls, the range of the year feature for modularity of the time series implementation later, and some summary statistics. The cleaned file was saved as a csv for loading into the API-genre-fetch file.

---

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




