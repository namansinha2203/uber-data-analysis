#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('uber_data_india_realistic.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['ride_date'] = pd.to_datetime(df['ride_date'])
df['ride_time'] = pd.to_datetime(df['ride_time'], format='%H:%M:%S').dt.time


# In[7]:


df['ride_date'] = pd.to_datetime(df['ride_date'], errors='coerce')

# Check for any conversion issues
if df['ride_date'].isnull().any():
    print("There are missing or invalid dates in the dataset.")
    print(df[df['ride_date'].isnull()])
else:
    print("All dates are valid.")

# Set 'ride_date' as the index
df.set_index('ride_date', inplace=True)

# Check the index
print(df.index)

# Resample the data by month and count the number of rides
monthly_rides = df.resample('M').size()

# Check the resampled data
print(monthly_rides)

# Plot the number of rides over time
plt.figure(figsize=(14, 7))
monthly_rides.plot()
plt.title('Number of Rides Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Rides')
plt.show()


# In[9]:


# 2. Ride status distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='ride_status')
plt.title('Ride Status Distribution')
plt.xlabel('Ride Status')
plt.ylabel('Count')
plt.show()


# In[11]:


# 3. Distribution of pickup locations
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='pickup_location', order=df['pickup_location'].value_counts().index)
plt.title('Distribution of Pickup Locations')
plt.xlabel('Pickup Location')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[12]:


# 4. Distribution of dropoff locations
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='dropoff_location', order=df['dropoff_location'].value_counts().index)
plt.title('Distribution of Dropoff Locations')
plt.xlabel('Dropoff Location')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[13]:


# 5. Average fare amount by vehicle type
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='vehicle_type', y='fare_amount')
plt.title('Average Fare Amount by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Average Fare Amount (INR)')
plt.show()


# In[14]:


# 6. Payment method distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='payment_method')
plt.title('Payment Method Distribution')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.show()


# In[15]:


# 7. Customer rating distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['customer_rating'], bins=5, kde=False)
plt.title('Customer Rating Distribution')
plt.xlabel('Customer Rating')
plt.ylabel('Frequency')
plt.show()


# In[16]:


# 8. Driver rating distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['driver_rating'], bins=5, kde=False)
plt.title('Driver Rating Distribution')
plt.xlabel('Driver Rating')
plt.ylabel('Frequency')
plt.show()


# In[17]:


# 9. Distribution of ride distances
plt.figure(figsize=(10, 5))
sns.histplot(df['distance_km'], bins=50, kde=True)
plt.title('Distribution of Ride Distances')
plt.xlabel('Distance (km)')
plt.ylabel('Frequency')
plt.show()


# In[18]:


# 10. Average fare amount by promo code
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='promo_code', y='fare_amount')
plt.title('Average Fare Amount by Promo Code')
plt.xlabel('Promo Code')
plt.ylabel('Average Fare Amount (INR)')
plt.show()


# In[19]:


# 11. Effect of weather conditions on ride duration
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='weather_condition', y='ride_duration_min')
plt.title('Effect of Weather Conditions on Ride Duration')
plt.xlabel('Weather Condition')
plt.ylabel('Ride Duration (min)')
plt.show()


# In[20]:


# 12. Effect of traffic conditions on ride duration
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='traffic_condition', y='ride_duration_min')
plt.title('Effect of Traffic Conditions on Ride Duration')
plt.xlabel('Traffic Condition')
plt.ylabel('Ride Duration (min)')
plt.show()


# In[21]:


# 13. Distribution of driver experience years
plt.figure(figsize=(10, 5))
sns.histplot(df['driver_experience_years'], bins=10, kde=True)
plt.title('Distribution of Driver Experience Years')
plt.xlabel('Years of Experience')
plt.ylabel('Frequency')
plt.show()


# In[22]:


# 14. Average customer rating by traffic condition
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='traffic_condition', y='customer_rating')
plt.title('Average Customer Rating by Traffic Condition')
plt.xlabel('Traffic Condition')
plt.ylabel('Average Customer Rating')
plt.show()


# In[23]:


# 15. Average driver rating by weather condition
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='weather_condition', y='driver_rating')
plt.title('Average Driver Rating by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Average Driver Rating')
plt.show()


# In[24]:


# 16. Popularity of promo codes
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='promo_code')
plt.title('Popularity of Promo Codes')
plt.xlabel('Promo Code')
plt.ylabel('Count')
plt.show()


# In[25]:


# 17. Analysis of cancellation reasons
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='cancellation_reason')
plt.title('Analysis of Cancellation Reasons')
plt.xlabel('Cancellation Reason')
plt.ylabel('Count')
plt.show()


# In[26]:


# 18. Effect of promo codes on fare amount
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='promo_code', y='fare_amount')
plt.title('Effect of Promo Codes on Fare Amount')
plt.xlabel('Promo Code')
plt.ylabel('Fare Amount')
plt.show()


# In[27]:


# 19. Ride distribution by vehicle type
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='vehicle_type')
plt.title('Ride Distribution by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Count')
plt.show()


# In[28]:


# 20. Average ride duration by vehicle type
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='vehicle_type', y='ride_duration_min')
plt.title('Average Ride Duration by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Average Ride Duration (min)')
plt.show()


# In[29]:


# 22. Effect of driver experience on customer rating
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='driver_experience_years', y='customer_rating', ci=None)
plt.title('Effect of Driver Experience on Customer Rating')
plt.xlabel('Years of Experience')
plt.ylabel('Average Customer Rating')
plt.show()


# In[30]:


# 23. Average fare amount by ride status
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='ride_status', y='fare_amount')
plt.title('Average Fare Amount by Ride Status')
plt.xlabel('Ride Status')
plt.ylabel('Average Fare Amount')
plt.show()


# In[31]:


# 24. Relationship between ride distance and fare amount for different vehicle types
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='distance_km', y='fare_amount', hue='vehicle_type', alpha=0.5)
plt.title('Relationship between Ride Distance and Fare Amount for Different Vehicle Types')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount')
plt.legend(title='Vehicle Type')
plt.show()


# In[32]:


# 25. Analysis of ride duration for different payment methods
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='payment_method', y='ride_duration_min')
plt.title('Ride Duration for Different Payment Methods')
plt.xlabel('Payment Method')
plt.ylabel('Ride Duration (min)')
plt.show()


# ## Statistical Analysis

# In[34]:


# Descriptive statistics for numerical columns
numerical_stats = df.describe()
numerical_stats


# In[39]:


# Summary statistics for categorical columns
categorical_summary = df.describe(include=['object', 'category'])
categorical_summary


# In[42]:


# Average fare amount
average_fare = df['fare_amount'].mean()
print(f"Average Fare Amount: {average_fare:.2f} INR")

# Average distance
average_distance = df['distance_km'].mean()
print(f"Average Distance: {average_distance:.2f} km")

# Average ride duration
average_duration = df['ride_duration_min'].mean()
print(f"Average Ride Duration: {average_duration:.2f} minutes")


# In[43]:


# Average customer rating
average_customer_rating = df['customer_rating'].mean()
print(f"Average Customer Rating: {average_customer_rating:.2f}")

# Average driver rating
average_driver_rating = df['driver_rating'].mean()
print(f"Average Driver Rating: {average_driver_rating:.2f}")


# In[44]:


# Average ride duration by traffic condition
average_duration_by_traffic = df.groupby('traffic_condition')['ride_duration_min'].mean()
print("Average Ride Duration by Traffic Condition:")
print(average_duration_by_traffic)


# In[45]:


# Average ride duration by weather condition
average_duration_by_weather = df.groupby('weather_condition')['ride_duration_min'].mean()
print("Average Ride Duration by Weather Condition:")
print(average_duration_by_weather)


# In[ ]:




