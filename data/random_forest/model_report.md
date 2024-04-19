
    # Model Training and Evaluation Report

    ## Model Details
    - Model Type: Random Forest Regressor
    - Number of Estimators: 50
    - Random State: 42

    ## Training Data
    - Number of records: 10537
    - Features used: ['Bollinger Bands Mean', 'Consumer Price Index for All Urban Consumers: All Items', 'EMA', 'Gross Domestic Product', 'MACD', 'MACD Signal Line', 'RSI', 'Unemployment Rate', 'adjusted_close', 'volume', 'log_volume', 'normalized_ATR', 'normalized_OBV']

    ## Test Data
    - Number of records: 2635

    ## Cross-Validation
    - CV Scores (5-fold): [0.99735725 0.99805828 0.99804665 0.99739053 0.99394842]
    - Average CV Score: 0.9969602267342548

    ## Performance Metrics
    - Mean Squared Error (MSE): 1572.8576887455008
    - Root Mean Squared Error (RMSE): 39.65926989677824
    - R-squared (R2): 0.9963331286703511

    ## Feature Importance
                                                    Feature  Importance
                                                    EMA    0.336245
                                   Bollinger Bands Mean    0.301215
                                         normalized_ATR    0.132681
                                                   MACD    0.066861
                                       MACD Signal Line    0.051056
                                         adjusted_close    0.030359
                                                 volume    0.024226
                                             log_volume    0.020811
Consumer Price Index for All Urban Consumers: All Items    0.015544
                                         normalized_OBV    0.007187
                                      Unemployment Rate    0.005950
                                 Gross Domestic Product    0.005264
                                                    RSI    0.002602
    