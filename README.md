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
- [Dataset and Spotify API Genre Fetch](#dataset-and-spotify-api-genre-fetch)
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
### Dataset and Spotify API Genre Fetch

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
11. name: name of the song/track.
12. 



