from pydantic import BaseModel
from typing import List, Union
from datetime import datetime

class CreditScoresPredictionInput(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: Union[float, str]
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: Union[datetime, str]
    PricingStrategy: Union[float, str]

class CreditScoresPredictionOutput(BaseModel):
    PredictedCreditScores: Union[float, List[float]]