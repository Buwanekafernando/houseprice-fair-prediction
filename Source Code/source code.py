import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix


# First Preprocess by Savindi

path = "/content/house_prices.csv"
df = pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip")

# Drop unwanted columns
df.drop(columns=["Dimensions", "Plot Area"], inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert price into rupees
def money_to_rupees(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "").strip()
    if "Lac" in x:
        return float(x.replace("Lac", "").strip()) * 1e5
    elif "Cr" in x:
        return float(x.replace("Cr", "").strip()) * 1e7
    else:
        return pd.to_numeric(x, errors="coerce")

df["Amount(in rupees)"] = df["Amount(in rupees)"].map(money_to_rupees)

# Extract sqft from area
import re
def extract_sqft(value):
    if pd.isna(value):
        return np.nan
    match = re.findall(r"\d+", str(value).replace(",", ""))
    return float(match[0]) if match else np.nan

df["Carpet Area"] = df["Carpet Area"].map(extract_sqft)
df["Super Area"] = df["Super Area"].map(extract_sqft)

df["Bathroom"] = pd.to_numeric(df["Bathroom"], errors="coerce")
df["Balcony"] = pd.to_numeric(df["Balcony"], errors="coerce")

# Car Parking cleaning
df["Car Parking"] = (
    df["Car Parking"]
      .astype(str)
      .str.lower()
      .str.extract(r"(open|covered)", expand=False)
      .map({"open": 1, "covered": 2})
      .fillna(0)
      .astype("Int64")
)

# Drop if > 60% missing
df = df.loc[:, df.isnull().mean() < 0.6]

# Fill missing values
for col in df.select_dtypes(include="object"):
    df[col].fillna("Unknown", inplace=True)
for col in df.select_dtypes(include=["float64","int64"]):
    df[col].fillna(df[col].median(), inplace=True)

# Location categorization
super_urban = ["mumbai","bangalore","chennai","hyderabad","kolkata","pune","ahmedabad","new-delhi","delhi"]
super_urban_suburbs = ["thane","navi-mumbai","gurgaon","noida","kalyan","badlapur","mohali","vapi","palghar","panchkula"]
urban = ["nagpur","jaipur","lucknow","indore","bhopal","surat","vadodara","coimbatore","madurai","mangalore","mysore","nashik","jabalpur","jodhpur","kanpur","faridabad","ghaziabad","raipur","vijayawada","visakhapatnam","siliguri","trichy","tiruchirappalli","trivandrum","thiruvananthapuram","udaipur","aurangabad","belgaum","gwalior","guntur","guwahati","jamshedpur","rajahmundry","palakkad","thrissur","kochi","kozhikode","nellore"]
rural = ["agra","allahabad","prayagraj","bhiwadi","bhubaneswar","durgapur","haridwar","pondicherry","solapur","sonipat","vrindavan","udupi","shimla","satara","navsari","zirakpur"]

def categorize_location(loc):
    if pd.isna(loc):
        return "Unknown"
    l = str(loc).strip().lower()
    if any(city in l for city in super_urban):
        return "Super Urban"
    elif any(city in l for city in super_urban_suburbs):
        return "Super Urban Suburb"
    elif any(city in l for city in urban):
        return "Urban"
    elif any(city in l for city in rural):
        return "Rural / Small Town"
    else:
        return "Other"

df["location"] = df["location"].map(categorize_location)

category_map = {"Super Urban": 4,"Super Urban Suburb": 3,"Urban": 2,"Rural / Small Town": 1,"Other": 0,"Unknown": 0}
df["location"] = df["location"].map(category_map)

# Rename columns
df = df.rename(columns={
    "Price (in rupees)": "Price_per_sqrft",
    "location" : "Location",
    "Carpet Area": "Carpet_Area",
    "Super Area": "Super_Area",
    "Car Parking": "Car_Parking",
    "Amount(in rupees)": "Amount_in_rupees",
    "overlooking" : "Overlooking",
    "facing" : "Facing"
})

df.to_csv("/content/house_prices_cleaned.csv", index=False)

# ------------------------
# Further Preprocess by Buddhima
# ------------------------
df = pd.read_csv("/content/house_prices_cleaned.csv")
df.columns = df.columns.str.strip().str.replace(' ', '_')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

def remove_outliers(df, numeric_cols):
    df_cleaned = df.copy()
    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

df_clean = remove_outliers(df, numeric_cols)
df_clean.to_csv('cleaned_data.csv', index=False)

# ------------------------
# EDA by Buwaneka
# ------------------------
df_clean = pd.read_csv('cleaned_data.csv')

numerical_cols = ['Amount_in_rupees', 'Price_per_sqrft', 'Carpet_Area', 'Super_Area']
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_clean[col], kde=True, color='deeppink')
    plt.title(f'Distribution of {col}')
    plt.show()

categorical_cols = ['Status', 'Transaction', 'Furnishing', 'Facing', 'Ownership', 'Car_Parking', 'Bathroom', 'Balcony']
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    df_clean[col].value_counts().plot(kind='bar', color='deeppink')
    plt.title(f'Count Plot of {col}')
    plt.show()

# ------------------------
# Model Training by Praneepa
# ------------------------
df = pd.read_csv("cleaned_data.csv")

location_map = {"Colombo": 4, "Colombo Suburbs": 3, "Other Urban": 2, "Other Rural": 1}
df["Location"] = df["Location"].map(location_map)

furnish_map = {"Furnished": 3, "Semi-Furnished": 2, "Unfurnished": 1}
df["Furnishing"] = df["Furnishing"].map(furnish_map)

df["Car_Parking"] = df["Car_Parking"].astype(int)

features = ["Location", "Super_Area", "Carpet_Area", "Bathroom", "Furnishing", "Car_Parking", "Balcony"]
X = df[features]
y = df["Amount_in_rupees"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Regression MAE:", mean_absolute_error(y_test, y_pred))
print("Regression MSE:", mean_squared_error(y_test, y_pred))
print("Regression R2:", r2_score(y_test, y_pred))

# Fairness classification
def classify_fairness_regression(user_input, entered_price, model, tolerance=0.3):
    input_df = pd.DataFrame([user_input])
    fair_price = model.predict(input_df)[0]
    if entered_price > fair_price * (1 + tolerance):
        return "Overpriced", fair_price
    elif entered_price < fair_price * (1 - tolerance):
        return "Underpriced", fair_price
    else:
        return "Fair", fair_price

def generate_fairness_labels(actual_prices, predicted_prices, tol=0.3):
    labels = []
    for actual, fair in zip(actual_prices, predicted_prices):
        if actual > fair * (1 + tol):
            labels.append("Overpriced")
        elif actual < fair * (1 - tol):
            labels.append("Underpriced")
        else:
            labels.append("Fair")
    return labels

y_test_labels = generate_fairness_labels(y_test, y_pred, 0.3)

# Example user input
user_input = {"Location": 1, "Super_Area": 1500, "Carpet_Area": 1200, "Bathroom": 2, "Furnishing": 3, "Car_Parking": 1, "Balcony": 2}
entered_price = 800000
result, fair_price = classify_fairness_regression(user_input, entered_price, model)
print("Entered price is:", result)
print("Estimated fair price:", fair_price)
