# 5400finalproject

## Changes:
### packaged entire code 
debuged, fixed few features and algothrim 
### data collection: 
- added pagination
- add url scraping, able to abtain full content
### Data processing: 
- extend missing stock_price value (for dates withoutout end_price (weekends), filled with the last available price value) 
- Drop rows with no article content
### Modeling: 
- able to select the best model out of 4 models. 
- added evaluation and visualization, output stored in result and visualization file


.
├── data
│   ├── __pycache__
│   │   ├── preprocess.cpython-311.pyc
│   │   └── sentiment_feature_eng.cpython-311.pyc
│   ├── preprocess.py
│   ├── processed
│   │   ├── Feature_engineered_data.csv
│   │   └── Processed_merged_data.csv
│   ├── raw
│   │   ├── Financial_news.csv
│   │   ├── historical_stock_prices.csv
│   │   └── test.ipynb
│   └── sentiment_feature_eng.py
├── financial_news_sentiment
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── data_collection
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── get_news.cpython-311.pyc
│   │   │   ├── get_stock_price.cpython-311.pyc
│   │   │   └── utils.cpython-311.pyc
│   │   ├── data
│   │   │   └── raw
│   │   │       └── Financial_news.csv
│   │   ├── get_news.py
│   │   ├── get_stock_price.py
│   │   ├── preprocessed_financial_news.py
│   │   └── utils.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── stock_model.cpython-311.pyc
│   │   └── stock_model.py
│   └── preprocess
├── pipeline.py
├── requirement.txt
├── results
│   ├── models
│   │   └── stock_prediction_best_model.pkl
│   └── visualizations
│       ├── Logistic Regression_confusion_matrix.png
│       ├── Logistic Regression_roc_curve.png
│       ├── Random Forest_confusion_matrix.png
│       ├── Random Forest_feature_importance.png
│       ├── Random Forest_roc_curve.png
│       ├── SVC_confusion_matrix.png
│       ├── SVC_roc_curve.png
│       ├── XGBoost_confusion_matrix.png
│       ├── XGBoost_feature_importance.png
│       └── XGBoost_roc_curve.png
└── setup.py
