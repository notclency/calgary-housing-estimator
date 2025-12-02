# Reflections
Briefly answer the following questions. Point form and 1-2 sentences is fine.

## Why did you choose your selected dataset?

<!-- Mainly I'm just curious what interests people -->

- Recently, one of our members, Glenn, bought a new house, and in his search, he noticed a great variation in
  prices between neighborhoods in Calgary. This got us interested in trying to further learn for ourselves how variables like location,
  property size, number of bedrooms, and proximity to amenities affect housing prices. Also, we saw that this dataset has real-world 
  applications and could be used to predict housing prices in Calgary, so it felt more valuable to attempt, even though we 
  likely wouldn't actually create a legitimately useful model.

## What are some of your key observations from the data exploration process?

<!-- Refer to your notebook as desired -->

- Deciding how to best manage the skewed data distribution i.e, whether applying log transformations versus other scaling methods,
  as well as dealing with potential outliers. Besides that, one of the biggest challenges for us was determining
  which columns had the most correlation with the target column. Also, finding feature transformations and interactions (feature engineering)
  that would further improve the relationship with the target column. 

## Why did you choose to process the data the way you did?

<!-- Some reasonable justification here -->

- We started processing the data by finding the columns with the most correlation to the target column by plotting and seeing
  if I could visually spot relationships to the target column. Then, we started deriving new columns that could potentially have
  a stronger relationship to the target column for example LOG_LAND_SIZE in hopes of handling the skewness of the data.
  Also, we created new categorical columns, for example columns that had wide ranges, such as LAND_USE_DESIGNATION,
  we narrowed this new column by grouping, so we derived with rows like Residential, multi-Residential, Commercial,
  Industrial etc. Besides deriving new columns, the way we handled missing values was interesting, we explored columns
  with missing data and did a contextual manual impute. For example, if a land size was missing(NaN) or
  had a value of 0 (assuming it's a data entry error), we would replace this with the median of land size
  properties that had the same categorical properties. We also did this for missing Year of Construction values.
  For preprocessing, we made separate pipelines for numerical and categorical features. We used the standard scaler
  to normalize the numeric features, making sure that all continuous variables were on a measurable scale,
  which is suitable for training. For the categorical features, we dind't need to use a simpIe imputer to handle
  missing values because we manually replaced them and we used the OneHotEncoder to convert categorical
  variables into a format suitable for training

## What additional information do you wish you had?

<!-- The provided data is pretty limited. What information do you think would help you to make a more effective prediction? -->

- Building Characteristics such as # of rooms, # of stories, # of bathrooms, etc. would have been helpful in predicting the target column.
  Also, the condition of the property, the quality of the property, a factor to determine neighborhood & market data such as distance to school
  district ratings, distance to public transit, parks, hospitals, and retail centers, malls etc, also other factors that measured risk
  of natural disaster, prior sale prices etc. All these would have been helpful in predicting the target column.

## What was the hardest part of this assignment?

<!-- Anything goes! Tell me more about the challenges you faced -->
The most difficult part for us was definitely trying to collaborate by working separately (using separate branches and then creating 
pull requests for changes etc). Since this assignment didn't lend itself well to modularizing parts away, we found it easier to just 
hop on a Discord call while one person streamed their screen (if we were not all on campus). We could then discuss the changes and 
visuals we got from the data in real time. Doing this felt like we could iterate much faster. So, for almost all of the data exploration, 
we would just have a different person stream their screen while we were "exploring together". 