# California-Housing-Regression
Linear Regression project with feature engineering and data preprocessing ( My first Linear Regression Project)

A machine learning project that predicts housing prices in California
using Linear Regression, with a focus on feature engineering and
data preprocessing.

## Dataset

The California Housing dataset from `sklearn.datasets`, containing
20,640 samples and 8 features:

- **MedInc** — median income in block group
- **HouseAge** — median house age in block group
- **AveRooms** — average number of rooms per household
- **AveBedrms** — average number of bedrooms per household
- **Population** — block group population
- **AveOccup** — average number of household members
- **Latitude** — block group latitude
- **Longitude** — block group longitude
      
**Target:** Median house value (in hundreds of thousands of dollars)

## Project Structure
├── California_Housing.ipynb   # Main notebook
└── README.md

## Workflow

1. Data loading and exploration
2. EDA — distributions, scatter plots, correlation matrix
3. Outlier analysis
4. Feature engineering
5. Model training and evaluation (before vs after feature engineering)
6. Cross-validation

## Feature Engineering

The following transformations improved model performance:

- **AveOccup clipping** — capped at 10 to handle unrealistic outliers
- **Population log transformation** — reduced right skew
- **Rooms_per_Person** — new ratio feature (AveRooms / AveOccup)
- **Bedrooms_per_Room** — new ratio feature (AveBedrms / AveRooms)
- **LocationCluster** — KMeans clustering (5 clusters) on
  Latitude/Longitude to capture geographic patterns

## Results

| | Before Feature Engineering | After Feature Engineering |
|---|---|---|
| R² | 0.5758 | 0.6938 |
| MAE | 0.5332 | 0.4690 |
| RMSE | 0.7456 | 0.6484 |
| CV Mean R² | 0.6115 | 0.6724 |
| CV Std | 0.0065 | 0.0115 |

Feature engineering improved R² by **~12%**.

## Experiments

Several experiments were conducted during development:

- **SGDRegressor vs LinearRegression** — both produced nearly
  identical results, so LinearRegression was kept for simplicity
  and interpretability
- **AveOccup outlier handling** — three strategies were tested:
  keeping outliers as-is, removing rows where AveOccup > 10,
  and clipping at 10. All three gave close results because
  outliers represent a very small fraction of the 20,640 data
  points. Clipping was chosen as it preserves all data while
  keeping values realistic
- **Dropping Latitude and Longitude** — reduced R² from 0.69
  to 0.63, confirming that geographic location is a strong
  predictor of housing prices
- **Dropping AveBedrms** — had no impact on results

## Key Findings

- `MedInc` (median income) is the strongest predictor of
  housing prices with a coefficient of 0.74
- Geographic location (Latitude/Longitude) has a strong
  negative coefficient, confirming that location heavily
  influences price
- The engineered feature `Rooms_per_Person` (coefficient 0.50)
  became the third most important feature after feature engineering
- Cross-validation std of 0.0115 confirms the model is stable
  and not overfitting

## Technologies

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- sklearn
