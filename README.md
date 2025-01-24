# 10Academy-Kifiya-Week-6

Column Name	Definition
TransactionId	Unique transaction identifier on platform
BatchId	Unique number assigned to a batch of transactions for processing
AccountId	Unique number identifying the customer on platform
SubscriptionId	Unique number identifying the customer subscription
CustomerId	Unique identifier attached to Account
CurrencyCode	Country currency
CountryCode	Numerical geographical code of country
ProviderId	Source provider of Item bought
ProductId	Item name being bought.
ProductCategory	ProductIds are organized into these broader product categories
ChannelId	Identifies if customer used web,Android, IOS, pay later or checkout
Amount	Value of the transaction. Positive for debits from customer account and negative for credit into customer account
Value	Absolute value of the amount
TransactionStartTime	Transaction start time
PricingStrategy	Category of Xente's pricing structure for merchants
FraudResult	Fraud status of transaction 1 -yes or 0-No

```
mkdir -p .github/workflows dashboard/{data,logs} deployment/{app,logs} logs notebooks resources/{configs,data/{raw,processed,preprocessed,cleaned},encoders,models/checkpoints,scalers} screenshots/{dashboard,deployment} scripts/{data_utils,modeling,utils} src tests
```

```
touch requirement.txt .github/workflows/ci.yml deployment/app/__init__.py notebooks/initial_EDA.ipynb scripts/{data_utils,modeling,utils}/__init__.py scripts/__init__.py src/__init__.py tests/__init__.py
```