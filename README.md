# 10Academy-Kifiya-Week-6


# Building a Scalable Credit Scoring Model: A Data-Driven Approach

## Understanding Credit Risk and Progress on Model Development for Bati Bank

In an era where digital transactions are rapidly increasing, financial institutions must rely on robust **credit-scoring models** to assess a customer's ability to repay loans or participate in Buy-Now-Pay-Later (BNPL) services. This project focuses on developing a **scalable, efficient, and production-ready credit-scoring system** for Bati Bank, integrating **machine learning** and **feature engineering** techniques to optimize decision-making.

## Introduction to Credit Risk

Credit risk refers to the potential financial loss resulting from a borrower’s failure to fulfill their financial obligations, which primarily includes defaults on loans or credit repayments. For financial institutions like Bati Bank, assessing and managing credit risk is crucial for ensuring profitability and sustainability, especially when offering new services such as Buy-Now-Pay-Later (BNPL). As part of Bati Bank’s partnership with an e-commerce company, a major task is the development of a credit scoring model. This model assesses customers' risk levels when applying for loans in the BNPL service. 

In this blog, the foundational concepts of credit risk, based on key references, will be explored, and how the progress on the task of building a robust credit scoring model to better understand and manage this risk for this new product is going will also be discussed.

---

## **📌 Problem Statement**

The goal was to build a **fraud-resistant, data-driven credit scoring model** to power the BNPL service offered by Bati Bank. Given transactional data from customers, the model predicts the likelihood of a transaction being fraudulent or financially risky, helping businesses make **real-time lending decisions** with confidence.

### **Challenges Faced**
- Handling **high-cardinality categorical features** like `CustomerId`.
- Creating a scalable **RFMS feature set (Recency, Frequency, Monetary, Severity)** from transaction history.
- **Encoding categorical features** without introducing data leakage or bias.
- **Ensuring consistency** between training and inference during deployment.

---

## **🔍 Solution Approach**

To tackle these challenges, a structured and modularized approach was followed:

## Understanding Credit Risk: Key References

### 1. **Basic Overview of Credit Risk**
   Credit risk models serve to predict the likelihood that a borrower will default on their financial obligations. These models help financial institutions evaluate the potential losses before approving loans. There are various approaches to measuring credit risk, but most credit risk models include several components:
   
   - **The borrower’s creditworthiness**
   - **The conditions in the market and industry**
   - **The terms and conditions of the financial product offered**

   Understanding how these factors interact helps determine the risk associated with lending to a particular customer. Many credit risk models utilize statistical techniques such as logistic regression, decision trees, or machine learning models to analyze large datasets containing borrower characteristics.

### 2. **Key Reference Guidelines**
   The guidelines provided in references like [The World Bank](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf) and [Alternative Credit Scoring Approaches](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf) provide an in-depth view of best practices in risk modeling. Some critical elements discussed in these references are:
   
   - **The use of historical data to evaluate credit risk**, where factors like transaction history, spending patterns, and account behaviors come into play.
   - **Categorizing borrowers** into various risk categories (high-risk, low-risk) and assigning each category a numerical score (credit score).
   - **Legal and regulatory frameworks** around credit scoring, such as Basel II, which recommends guidelines on capital requirements and governance for banks dealing with credit risk.
  
   In simpler terms, credit scoring helps turn the complexities of financial decision-making into quantitative predictions of creditworthiness, which can be automated and used to help decide loan approvals.

### 3. **Advanced Approaches to Credit Risk Assessment**
   Credit risk assessment has advanced significantly with new methodologies and tools. Statistical tools like Weight of Evidence (WoE) and Information Value (IV) allow analysts to better interpret factors influencing credit risk. These advanced methods create more insightful and accurate scoring models compared to traditional approaches.
   
   According to a comprehensive piece on [Towards Data Science](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03), data exploration, feature engineering, and statistical learning are used to predict default risk efficiently. Features such as user transaction behavior, geographical data, product type, fraud history, and more are commonly used predictors.

## Task 1: Credit Risk in the Context of Bati Bank's Buy-Now-Pay-Later (BNPL) Service

Understanding the core elements of credit risk is critical when constructing a predictive model for evaluating the creditworthiness of customers on the BNPL platform. The key objectives here revolve around defining proxy variables that signal the risk of default and selecting observable features strongly correlated with default. Key insights gathered from Task 1 will help construct this predictive model.

The project’s goals for **Task 1 – Understanding Credit Risk** involve:

1. **Defining Risk Proxy Variables**: Identifying what makes someone high or low risk; in this case, looking at observable financial behaviors, like transaction history, debt utilization, payment history, and potentially, whether fraud or disputes occur in transaction records.
   
2. **Feature Selection**: Aiming to select features such as `Amount`, `TransactionStartTime`, `PricingStrategy`, and `FraudResult` as likely predictors for risk. This decision is based on the premise that transactions (as either debits or credits into the user account) provide valuable information about a customer's financial stability.

3. **Default Estimation**: By establishing a boundary between good and bad credit behaviors (through proxy risk classification), Bati Bank can begin categorizing users. This approach aligns with the research in [Risk Officer](https://www.risk-officer.com/Credit_Risk.htm), which emphasizes simple, data-driven classification techniques for evaluating borrower creditworthiness.

As of now, progress on Task 1 in constructing the credit risk model includes:

- **Exploration of Data**: Will be starting to gather transaction-level data to identify customer behaviors like spending patterns, the frequency of loan repayments, and fraud histories.
- **Binning and WoE**: The initial preparations for building **Weight of Evidence (WoE)** models are in progress. This will help transform raw data into easily interpretable variables that are used to predict creditworthiness.

## Progress and Insights on Task 2: Exploratory Data Analysis (EDA)

Substantial progress has been made with Task 2, focusing on the **Exploratory Data Analysis** (EDA) and cleaning the data for the next steps of feature engineering. Key steps completed so far include:

- **Summary Statistics**: Gaining an understanding of the data distribution and correlation between numerical features, specifically identifying outliers in monetary transactions (`Amount`), payment behavior (`Value`), and transaction types (`PricingStrategy`).
  
- **Missing Data Analysis**: Identifying columns with missing values like `PricingStrategy`, `FraudResult`, and `Value`. This helps define imputation strategies.
  
- **Visualization and Plotting**: Using advanced visualization techniques (e.g., histograms, box plots, and bar plots) to understand the distribution of numerical and categorical features, such as `CountryCode`, `ProductCategory`, and `TransactionStartTime`.

**Data Cleaning and Feature Generation** has begun with initial steps for imputing missing values and creating aggregate features (e.g., total transaction amount, transaction count per user). These insights lay the foundation for model readiness.

Task 1 has helped lay a solid foundation for the credit risk model development. By using the references and expert guidelines, a better understanding of how customer behaviors affect their creditworthiness has been achieved. As a result, the initial step of defining risk categories through proxy variables is clear.

### **📊 Data Processing & Feature Engineering**

1. **Key Features Considered**
   - `Recency`: How recently the customer made a transaction.
   - `Frequency`: How often they transact.
   - `Monetary`: Total value of transactions.
   - `Severity`: Number of fraudulent transactions.
   - WoE Encoding on categorical fields like `ProductId`, `ChannelId`, and `ProviderId`.

2. **Feature Transformation Pipelines**
   - **Custom WoE Transformer**: Maps categorical variables to **Weight-of-Evidence** (WoE) scores.
   - **RFMS Feature Transformer**: Computes Recency, Frequency, and Monetary variables dynamically.
   - **Scaling and Normalization**: Standardizes data to optimize ML model performance.
     
### **⚙️ Model Development**

Multiple machine learning models were trained, evaluating them for accuracy and interpretability. The models tested included:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier** (best performance)
- **LightGBM for Optimization**

> **Evaluation Metrics Used:** F1-score, AUC-ROC, and Precision-Recall curves to balance fraud detection with minimal false positives.

### **🚀 Deployment and Inference Consistency**
To ensure the **preprocessing pipeline is applied consistently** across training and real-time inference:
- **Scikit-learn Pipelines** encapsulated feature processing for seamless transformation.
- **Preprocessing + Model Bundling** using Joblib (`preprocessing_pipeline.pkl`, `full_pipeline_model.pkl`).
- **FastAPI-based ML Service**
  - Real-time API accepts raw transaction data.
  - Automatically applies **the same feature engineering pipeline**.
  - Predicts creditworthiness in milliseconds.

---

## **🔑 Key Takeaways**

✅ **Feature engineering is as important as the model itself.** RFMS and WoE transformations significantly boosted predictive accuracy.

✅ **Pipeline-based processing ensures reproducibility and stability.** Every prediction during inference used the same transformations applied during training.

✅ **API-based model deployment integrates ML into financial decision-making.** Predictions can be made on real-time transactions with high efficiency.

✅ **Interpretability remains critical in FinTech applications.** Feature importance analysis helped justify lending decisions.

---

## **💡 Final Thoughts**

Building a credit-scoring model is not just about selecting the best algorithm. The real power lies in **feature engineering, handling data consistency, and seamless deployment.** With this scalable, modular approach, financial institutions can **accurately assess risk in real-time** while minimizing fraud and optimizing their BNPL services.
