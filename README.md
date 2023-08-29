![banner](images/hotel_incentive.png)

![Tools](https://img.shields.io/badge/Tool-Python-lightgrey)
![Type of ML](https://img.shields.io/badge/Method-Supervised_ML-red)
![GitHub last commit](https://img.shields.io/github/last-commit/duynlq/scraped-reviews-customer-churn-prediction)
![GitHub repo size](https://img.shields.io/github/repo-size/duynlq/scraped-reviews-customer-churn-prediction)

Badge [source](https://shields.io/)

# Key findings: UNDER CONSTRUCTION

## Author
- [@duynlq](https://github.com/duynlq)

## Business Problem

Increasing customer retention and optimizing marketing resource allocation are critical challenges for the hospitality industry. I aim to develop an optimal predictive model that identifies customers at a high risk of churning from our services. Since we can sustain significant costs in incentivized marketing for churn customers, my primary focus is on minimizing the number of customers wrongly identifies as churn. By achieving a high rate of predicting customers who will actually churn, we can ensure that our marketing efforts are targeted towards the proper audience, and therefore maximizing our customer retention rates and reducing unnecessary marketing expenses.

## Data Source

- Our scope of reference is the first *page of reviews* scraped for the first 10 *pages of hotels* within Austin, TX on August 2023, sorted by Best Value. All first pages of reviews has a maximum of 10 reviews, but not all hotels in the scraped data will have all 10. It can be assumed that lesser Best Value hotels has a lesser chance of being reviewed.
- 2013 unique reviews were scraped.
- 294 unique hotel names were scraped.

## Quick EDA
![reviews distribution](images/scraped_reviews_distribution.png)
- Customers are more likely to put trust in a hotel based on its recent reviews. 59% of my scraped reviews are from 2023, despite having a time range from December 2017 to August 2023.

![class distribution](images/class_distribution.png)
- Class imbalance in the ratings distribution can negatively impact the accuracy of my models.

## Quick glance at the results
TODO

## Lessons learned and recommendation**

## Limitation and what can be improved

## Repository Structure

## Methods
- Web-scraping: BeautifulSoup
- Data Validation
- Exploratory Data Analysis (EDA)
- Natual Language Processing (NLP): NLTK, Vectorizing, Tokenizing
- Supervised Machine Learning: Logistic Regression, Random Forest
- A/B Testing
  
## Tech Stack
- Python
- SQL
- Tableau
