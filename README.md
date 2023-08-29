![banner](images/hotel_incentive.png)

![Tools](https://img.shields.io/badge/Tool-Python-lightgrey)
![Type of ML](https://img.shields.io/badge/Method-Supervised_ML-red)
![GitHub last commit](https://img.shields.io/github/last-commit/duynlq/scraped-reviews-customer-churn-prediction)
![GitHub repo size](https://img.shields.io/github/repo-size/duynlq/scraped-reviews-customer-churn-prediction)

Badge [source](https://shields.io/)

# Key findings: ?

## Author
- [@duynlq](https://github.com/duynlq)

## Business Problem

Increasing customer retention and optimizing marketing resource allocation are critical challenges for the hospitality industry. I aim to develop an optimal predictive model that identifies customers at a high risk of churning from our services. Since we can sustain significant costs in incentivized marketing for churn customers, my primary focus is on minimizing the number of customers wrongly identifies as churn. By achieving a high rate of predicting customers who will actually churn, we can ensure that our marketing efforts are targeted towards the proper audience, therefore maximizing our customer retention rates and reducing unnecessary marketing expenses.

## Data Source

- [American Craft Beers](https://forecasters.org/resources/time-series-data/m3-competition/](https://www.kaggle.com/datasets/nickhould/craft-cans))

## Methods

- ARIMA (differencing all realizations)
- ARIMA (differencing via Cochrane–Orcutt estimation)
- Holt-Winter's Additive
- Holt-Winter's Multiplicative
- [Theta](https://www.sciencedirect.com/science/article/abs/pii/S0169207000000662)
- [CES](https://onlinelibrary.wiley.com/doi/full/10.1002/nav.22074) (complex exponential smoothing)
- [ES](https://www.sciencedirect.com/science/article/abs/pii/S0169207001001108) (exponential smoothing state space model)
- [MLP](https://kourentzes.com/forecasting/2019/01/16/tutorial-for-the-nnfor-r-package/)
- [DeepAR](https://www.sciencedirect.com/science/article/pii/S0169207019301888)
- [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735)
  
## Tech Stack
- R (refer to [here](https://github.com/tiddles585/Capstone/blob/duy_branch/R/Functions.R) for the libraries used)
- Python (used to implement LSTM and DeepAR)

## Quick glance at the results

Correlation between the features.

![heatmap](assets/heatmap.png)

Confusion matrix of gradient boosting classifier.

![Confusion matrix](assets/confusion_matrix.png)

ROC curve of gradient boosting classifier.

![ROC curve](assets/roc.png)

Top 3 models (with default parameters)

| Model     	                | Recall score 	|
|-------------------	        |------------------	|
| Support vector machine     	| 88% 	            |
| Gradient boosting    	        | 90% 	            |
| Adaboost               	    | 79% 	            |



- **The final model used for this project: Gradient boosting**
- **Metrics used: Recall**
- **Why choose recall as metrics**:
  Since the objective of this problem is to minimize the risk of a credit default, the metrics to use depends on the current economic situation:

  - During a bull market (when the economy is expanding), people feel wealthy and are employed. Money is usually cheap, and the risk of default is low because of economic stability and low unemployment. The financial institution can handle the risk of default; therefore, it is not very strict about giving credit. The financial institution can handle some bad clients as long as most credit card owners are good clients (aka those who pay back their credit in time and in total).In this case, having a good recall (sensitivity) is ideal.

  - During a bear market (when the economy is contracting), people lose their jobs and money through the stock market and other investment venues. Many people struggle to meet their financial obligations. The financial institution, therefore, tends to be more conservative in giving out credit or loans. The financial institution can't afford to give out credit to many clients who won't be able to pay back their credit. The financial institution would rather have a smaller number of good clients, even if it means that some good clients are denied credit. In this case, having a good precision (specificity) is desirable.

    ***Note***: There is always a trade-off between precision and recall. Choosing the right metrics depends on the problem you are solving.

    ***Conclusion***: Since the time I worked on this project (beginning 2022), we were in the longest bull market (excluding March 2020 flash crash) ever recorded; we will use recall as our metric.


 **Lessons learned and recommendation**

- Based on this project's analysis, income, family member headcount, and employment length are the three most predictive features in determining whether an applicant will be approved for a credit card. Other features like age and working employment status are also helpful. The least useful features are the type of dwelling and car ownership.
- The recommendation would be to focus more on the most predictive features when looking at the applicant profile and pay less attention to the least predictive features.

## Limitation and what can be improved

- Combine this model with with a regression model to predict how much of a credit limit an applicant will be approved for.
- Hyperparameter tuning with grid search or random search.
- Better interpretation of the chi-square test
- Retrain the model without the least predictive features

## Repository Structure
```

├── assets
│   ├── confusion_matrix.png                      <- confusion matrix image used in the README.
│   ├── gif_streamlit.gif                         <- gif file used in the README.
│   ├── heatmap.png                               <- heatmap image used in the README.
│   ├── Credit_card_approval_banner.png           <- banner image used in the README.
│   ├── environment.yml                           <- list of all the dependencies with their versions(for conda environment).
│   ├── roc.png                                   <- ROC image used in the README.
│
├── datasets
│   ├── application_record.csv                    <- the dataset with profile information (without the target variable).
│   ├── credit_records.csv                        <- the dataset with account credit records (used to derive the target variable).
│   ├── test.csv                                  <- the test data (with target variable).
│   ├── train.csv                                 <- the train data (with target variable).
│
│
├── pandas_profile_file
│   ├── credit_pred_profile.html                  <- exported panda profile html file.
│
│
├── .gitignore                                    <- used to ignore certain folder and files that won't be commit to git.
│
│
├── Credit_card_approval_prediction.ipynb         <- main python notebook where all the analysis and modeling are done.
│
│
├── LICENSE                                       <- license file.
│
│
├── cc_approval_pred.py                           <- file with the model and streamlit component for rendering the interface.
│
│
├── README.md                                     <- this readme file.
│
│
├── requirements.txt                              <- list of all the dependencies with their versions(used for Streamlit).

```
