# Melbourne Housing Price Predictor

A simple machine learning project that predicts house prices in Melbourne, Australia based on property details like rooms, location, and land size.

## Dataset

Melbourne Housing Snapshot dataset from Kaggle (originally scraped from Domain.com.au by Tony Pino).
Link: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot

It has 13,580 property records with details like price, rooms, location, land size, and sale date.

## What I Did

**1. Data Cleaning**
- Checked missing values in `BuildingArea`, `YearBuilt`, `CouncilArea`, and `Car`.
- Dropped rows with missing `Car` values.
- Filled `CouncilArea` with mode, and `YearBuilt` / `BuildingArea` with median.
- Converted `Date` column to proper datetime format.

**2. Outlier Detection**
- Used the IQR method to check outliers in `Price`.
- Visualized them with a boxplot.

**3. EDA (Exploratory Data Analysis)**
- Checked price distribution (it's right-skewed).
- Compared price with rooms, bathrooms, location, and type of property.
- Built a correlation heatmap to see which features affect price the most.
- Rooms, Bedroom2, and Bathroom turned out to be the strongest price drivers.

**4. Feature Engineering**
- Extracted `SaleYear` and `SaleMonth` from the date.
- Dropped `Address` (too many unique values, not useful for prediction).
- One-hot encoded categorical columns.

**5. Model Training**
- Trained 6 models: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, XGBoost.
- Compared them using MAE, RMSE, and R².
- Picked the top 2 performing models and tuned them further with GridSearchCV.
- Saved the best model using pickle (along with the scaler and feature list).

**6. Deployment**
- Built a Streamlit web app (`app.py`) where a user can enter property details and get a predicted price instantly.

## How to Run

1. Install the requirements:
```
pip install pandas numpy scikit-learn xgboost streamlit seaborn matplotlib
```

2. Make sure these files are in the same folder:
   - `app.py`
   - `melbourne_house_price_model.pkl`
   - `melb_data.csv`

3. Run the app:
```
streamlit run app.py
```

4. Open the link Streamlit gives you (usually `http://localhost:8501`).

## Files in this Repo

| File | Description |
|---|---|
| `Melbourne_Housing_Snapshot.ipynb` | Full notebook: cleaning, EDA, model training, tuning |
| `app.py` | Streamlit app for predictions |
| `melb_data.csv` | Dataset |
| `melbourne_house_price_model.pkl` | Saved trained model |

## Result

Best model selected based on Test R² score after hyperparameter tuning. Full comparison and metrics are in the notebook.

## Notes

This is a learning project built to practice the full data science workflow — cleaning, EDA, modeling, and deployment.
