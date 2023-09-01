![banner](images/hotel_incentive.png)

![Tools](https://img.shields.io/badge/Tools-Python,_SQL,_Tableau-yellow)
![Methods](https://img.shields.io/badge/Methods-Webscraping,_NLP,_Supervised_ML-red)
![GitHub last commit](https://img.shields.io/github/last-commit/duynlq/scraped-reviews-customer-churn-prediction)
![GitHub repo size](https://img.shields.io/github/repo-size/duynlq/scraped-reviews-customer-churn-prediction)

Badge [source](https://shields.io/)

# Key findings: UNDER CONSTRUCTION

## Author
- [@duynlq](https://github.com/duynlq)

## Business Problem

Increasing customer retention and optimizing marketing resource allocation are critical challenges for the hospitality industry. I aim to develop an optimal predictive model that identifies customers at a high risk of churning from our services. Since we can sustain significant costs in incentivized marketing for churn customers, my primary focus is on minimizing the number of customers wrongly identifies as churn. By achieving a high rate of predicting customers who will actually churn, we can ensure that our marketing efforts are targeted towards the proper audience, and therefore maximizing our customer retention rates and reducing unnecessary marketing expenses.

## Data Source

- Our scope of reference is the **_first page of reviews_** scraped from the **_first 10 pages of hotels_** within Austin, TX on August 2023, sorted by Best Value. All first pages of reviews has a maximum of 10 reviews, but not all hotels in the scraped data will have all 10. It can be assumed that lesser Best Value hotels has a lesser chance of being reviewed.
- 2013 unique reviews were scraped.
- 294 unique hotel names were scraped.
- Customers are more likely to put trust in a hotel based on its recent reviews. 59% of my scraped reviews are from 2023, despite having a time range from December 2017 to August 2023.
  ![reviews distribution](images/scraped_reviews_distribution.png)

## Data Validation & Cleaning
- **_Duplicate Removal:_** Removing duplicate reviews is a must to avoid introducing bias to our models. Since the rows of reviews (independent variable) are our only interest, duplicates were checked and none were found.
- **_Missing Data:_** Among all columns, missing values was only found in the "date_stayed" column, but since this variable was only used for visualizations and not in our models, it was left as is.
- **_Text Cleaning:_** The reviews that are stored in my local database, which embody both the title and the "Read More" section of the reviews, are extracted from the "span" HTML tag and processed by the following steps:
  - Convert to lowercase.
  - Tokenize by isolating one or more word characters appeared in a row (alphanumberic characters plus underscore "_").
  - Remove English stop words from "stop_words" (174 stop words) and "nltk" (179 stop words) Python libraries.
  - Remove punctuations.

## Preprocessing
- **_Text Processing:_** The following steps are necessary for building a quality vocabulary from the cleaned corpus, or collection of texts, before converting it into numerical features that are most suitable for machine learning algorithms.
  - Tokenize by isolating one or more uppercase and lowercase letters appeared in a row (AAAHH).
  - Tokenize by isolating one or more lowercase letters immediately following an apostrophe "'" appeared in a row (I'mm).
  - Assign part-of-speech tags from "nltk" to the processed words so far, based on both their definitions and their contexts.
  - Exclusively includes nouns, adjectives, verbs, and adverbs from WordNet with a custom function. "Early [research papers] (1) has focused on using adjectives such as ‘good’ and ‘bad’ and adverbs like ‘terrifically’ and ‘hatefully’ as the important indicators of sentiment (2). Intuitively, this is what we would expect of an opinionated document. However, later [research papers] (3) also suggests that other parts of speech such as verbs and even nouns (4) could be valuable indicators of sentiment."
  - Lemmatize by replacing words with their root form (caring -> care, feet -> foot, striped -> strip) to reduce dimensionality and maintain consistency. The latter example is one problem that is yet to be taken care of.
  - IDEA: Build app that showcases the cleaned and processed reviews live in action prior for modeling.
- **_Train/test Split:_** 70% of the reviews (independent variable) and our target churn/non-churn (dependent variable) is used to train our models, where 30% is used to validate/test them. This ratio is a common practice in data science.
- **_Vectorization:_** TfidfVectorizer() is used to omit words that both appear in more than 10% and less than 5% of the cleaned and processed reviews, before converting then into numerical features which embody a matrix of TF-IDF features (Term Frequency-Inverse Document Frequency). As said earlier, this format is most suitable for machine learning algorithms.
- **_Principle Component Analysis (PCA):_**
  - PCA is used to reduce the number of TF-IDF features extracted from the reviews (dimensionality reduction), while getting rid of collinear features which will end up in a single PCA component.
  - Vectorized reviews are scaled from 0 to 1 via MinMaxScaler() prior to dimensionality reduction via PCA(), since 0 to 1 scaling is a must for PCA. Not all models will utilize PCA.
- The vectorized and features-reduced reviews will now be referred as the processed reviews.

## Introduction of The Compared Models
- **_Logistic Regression_**
  - **_Feature Importance:_** Provides straightforward interpretation, namely importance features stating which aspects of hotel reviews influence the likelihood of customer churn.
  - **_Commonly Used Model:_** It is a simple yet effective approach to modeling binary response variables (in this case churn vs non-churn) and can serve as a solid baseline model to compare against our other models.
  - **Hyperparameter Tuning:_**  Tuning with cross-validation was kept simple with 5-fold cross-validation and grid search with class_weight='balanced', C from 0.0 to 1.0, max_iter=100, and penalty between 'none' and 'l2'.
- **_Random Forest_**
  - **_Nonlinear Relationships:_** Can also provide features importance as well as capture nonlinear relationships between features effectively. In the context of hotel reviews, this model type can better handle the complex relationships of sentimental values within the reviews.
  - **_PCA Warning:_** Model was not trained with the processed reviews, since it does not perform well when features are monotonic transformations of other features, making the forest trees less independent from each other.
  - **Hyperparameter Tuning:_** Tuning with cross-validation was first done with 5-fold cross-validation and grid search with max_features and n_estimators, then criterion and max_depth, then min_sample_leaf and min_sample_split, and finally class_weight.
- **_Decision Tree_**
  - **_Customer Segmentation:_** Can also provide features importance as well as naturally divide the data into segments based on feature values, which is useful for identifying specific groups of customers who are more likly to churn based on their reviews.
  - **_Highly Interpretable:_** The dividing process can be visualized to be showcased to a non-technical audience.
  - **Hyperparameter Tuning:_** Tuning with cross-validation was first done with 5-fold cross-validation and grid search with max_features and max_leaf_nodes, then criterion and max_depth, then min_sample_leaf and min_sample_split, and finally class_weight.

## Distribution of Ratings & Target Labeling
  ![class distribution](images/class_distribution.png)
  - **_Class Imbalance_:** Imbalance in the ratings distribution can negatively impact the accuracy of the models. Converting the target variable to a binary variable, in this case churn or non-churn, will help alleviate this class imbalance.
  - **_Non-churn:_** To aid this decision for target labeling, it can be assumed that there is strong evidence in reviews starred 4 and 5 that the customer in question will most likely book a room again or recommend it to friends or family, while giving constructive criticism for some minor negative experience. These customers will be put in the non-churn category.
  - **_Churn:_** Likewise, there is strong evidence in reviews starred 3, 2 and 1 that the customer in question will either state that the hotel is average compared to others, gave largely negative comments to a majority of their experience, and frankly denies to recommend the hotel to anyone based on their entire experience. These customers will be put in the churn category.
 
## Class Weight
- THIS IS WRONG, NEEDS REINSTATEMENT, CANNOT USE class_weight FOR PRECISION-FOCUSED, Since the class balance is still imbalanced (roughly 1:2 for churn:non-churn), we can simply use the tuning hyperparameter "class_weight" available in all our compared models, instead of considering techniques like resampling or SMOTE. Such techniques **_will_** be necessary when building models that predict multiple classes, such as the five unique ratings apparent in the reviews.
- As mentioned earlier, by achieving a high rate of predicting customers who will actually churn, or True Positives (TP), we can ensure that our marketing efforts are targeted towards the proper audience. Since we can sustain significant costs in incentivized marketing for churn customers, my primary focus is on minimizing the number of customers wrongly identifies as churn, or False Positives (FP). Naturally, this aligns with the need to strive for a higher precision.
    ![precisionvsrecall](images/precision_vs_recall.png)
- Therefore, "class_weight" will be individually 

## Results
| Model    | Accuracy | Precision | Recall |
| :-------- | -------: | --------: | -------: |
| Logistic Regression       | 86% | 89% | 67% |
| Logistic Regression (PCA) | 84% | 78% | 75% |
| Random Forest             | 86% | 85% | 71% |
| Decision Tree             | 78% | 79% | 50% |
| Decision Tree (PCA)       | 81% | 74% | 70% |

![ROC_curves](images/ROC_curves.png)

## Final Verdict
- Random Forest, as of development on 9/1/2023, was able to achieve the best accuracy and precision, where the other compared models could not.

## Limitations & Future Improvements
- **_Validation Tests:_**
  - PCA is known to perform worse for datasets with more features than samples. The number of "features" created by preprocessing is far too greater than the number of review samples scraped, which is only around 2000. Between "Logistic Regression" and "Logistic Regression (PCA)", with the ability to scrape a much larger number of recent reviews, we should see better performance for the latter.
  - Random Forest also typically loses performance when there are more features than samples, therefore the ability to scrape more reviews is needed.
  - However, Decision Tree is prone to overfitting with addition of more data
- **_Modeling:_** Models compared can be extended to K Neared Neighbors (KNN), Support Vector Machines (SVM), and Naive Bayes, all of which can be implemented as binary classifiers.
- **_Feature Engineering:_** Features can be extended to include the review length, calculated sentiment scores from each review, one-hot coded topics extracted from each review.

## Repository Structure

## References
- (1) https://dl.acm.org/doi/10.3115/976909.979640
- (2) https://www.sciencedirect.com/science/article/pii/S0167923612001340#bb0070
- (3) https://dl.acm.org/doi/10.1145/1099554.1099714
- (4) https://dl.acm.org/doi/10.3115/1119176.1119180
