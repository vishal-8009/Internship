import joblib
import pandas as pd
hpp=joblib.load("house_rent_prediction.pkl")
encoder=hpp["encoder"]
scaler=hpp["scaler"]
model=hpp["model"]
new_house = pd.DataFrame({
'BHK': [2],
'Size': [1200],
'Area Type': ['Super Area'],
'City': ['Delhi'],
'Furnishing Status': ['Semi-Furnished'],
'Tenant Preferred': ['Family'],
'Bathroom': [2],
'Point of Contact': ['Contact Owner']
})

new_encoded = encoder.transform(new_house) 
new_scaled = scaler.transform(new_encoded) 
prediction = model.predict(new_scaled) 
print("Predicted Rent:", prediction[0])