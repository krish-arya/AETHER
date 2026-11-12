# PROJECT PRD
# Project AETHER
### An Integrated GAN–Reinforcement Learning–LLM Framework for Quantitative Market Intelligence and Portfolio Optimization

**Document Type:** Product Requirements Document (PRD) + Technical Specification  
**Version:** 1.0  
**Target Market:** National Stock Exchange of India (NSE)  
**Product Category:** AI / Quantitative Finance / Algorithmic Research  
**Primary Users:** Quantitative researchers, retail investors, portfolio managers and financial technology developers  
**Development Approach:** Modular, research-driven, API-first  
**Initial Deployment:** Local development → Cloud deployment → Paper trading

---

# 1. Executive Summary

## 1.1 Product Vision

Build an intelligent quantitative financial research and portfolio management platform that combines three major AI paradigms:

1. **Generative Adversarial Networks (GANs):** To generate realistic synthetic financial market scenarios, model market regimes and improve the robustness of training environments.
2. **Reinforcement Learning (RL):** To learn optimal portfolio allocation and trading policies through interaction with historical and simulated market environments.
3. **Large Language Models (LLMs):** To interpret financial news, corporate announcements, company fundamentals, macroeconomic events and market sentiment, converting unstructured information into structured intelligence.

The system should integrate these components into a unified architecture capable of:

- Processing historical and real-time market data.
- Identifying market regimes.
- Understanding fundamental and technical market conditions.
- Extracting actionable intelligence from financial news.
- Generating synthetic market scenarios.
- Training reinforcement learning agents.
- Optimizing portfolio allocation under risk constraints.
- Evaluating strategies through realistic backtesting.
- Providing explainable portfolio recommendations.
- Monitoring portfolio performance through an interactive dashboard.

The platform must be designed as a research and decision-support system rather than an automated profit-generation tool.

## 1.2 Core Problem

Traditional retail trading systems generally suffer from several limitations:

- Technical indicators operate primarily on historical numerical data.
- Traditional ML models often optimize price prediction rather than portfolio-level outcomes.
- Historical market datasets provide limited exposure to rare market events.
- Financial news and corporate announcements are often analyzed separately from quantitative trading systems.
- Many RL trading systems lack realistic transaction costs, liquidity constraints and robust out-of-sample evaluation.
- Most systems provide limited explanations for their allocation decisions.

Project AETHER addresses these problems by integrating generative modeling, sequential decision-making and language-based financial intelligence.

## 1.3 Product Objectives

| Objective | Expected outcome |
|---|---|
| Market intelligence | Unified quantitative and qualitative analysis |
| Scenario generation | Synthetic market environments |
| Portfolio optimization | Risk-aware RL allocation |
| News intelligence | Structured financial event analysis |
| Explainability | Traceable decision rationale |
| Research evaluation | Reproducible quantitative experiments |
| Deployment | Interactive research and paper-trading platform |

---

# 2. Product Scope

## 2.1 In Scope

The first version must support:

- NSE-listed equities.
- Daily OHLCV market data.
- Historical company fundamentals where legally available.
- Financial news and corporate announcements.
- Technical indicators.
- Market regime classification.
- Synthetic market data generation.
- GAN-based scenario generation.
- RL-based portfolio optimization.
- LLM-based news and event intelligence.
- Multi-model feature fusion.
- Historical backtesting.
- Portfolio risk analytics.
- Paper trading.
- Interactive dashboards.
- Experiment tracking and model versioning.

## 2.2 Out of Scope for Initial MVP

The following should not be implemented initially:

- Automated execution using real brokerage accounts.
- High-frequency trading.
- Intraday market-making.
- Options and futures strategies.
- Unrestricted autonomous financial transactions.
- Unverified social-media trading signals.
- Guaranteed return or price predictions.

The architecture should allow these capabilities to be explored in future versions.

---

# 3. System Architecture

The system should follow a modular, loosely coupled architecture.

Each major component must operate independently and communicate through well-defined interfaces.

## 3.1 High-Level Architecture

```text
                   DATA INGESTION LAYER
                            |
          +-----------------+-----------------+
          |                 |                 |
      Market Data       Financial News     Fundamentals
          |                 |                 |
          +-----------------+-----------------+
                            |
                    DATA PROCESSING
                            |
          +-----------------+-----------------+
          |                 |                 |
     Quant Features    Text Processing    Data Validation
          |                 |                 |
          +-----------------+-----------------+
                            |
                 MARKET INTELLIGENCE LAYER
                            |
          +-----------------+-----------------+
          |                                   |
      GAN ENGINE                          LLM ENGINE
          |                                   |
 Synthetic Scenarios                  News Intelligence
 Regime Conditioning                 Event Extraction
 Stress Scenarios                    Sentiment Analysis
          |                                   |
          +-----------------+-----------------+
                            |
                   FEATURE FUSION
                            |
                  MARKET STATE BUILDER
                            |
                     RL ENGINE
                            |
                 PORTFOLIO ALLOCATION
                            |
                  RISK MANAGEMENT
                            |
              +-------------+-------------+
              |             |             |
           Backtest      Paper Trading   Analytics
              |             |             |
              +-------------+-------------+
                            |
                     USER DASHBOARD
```

## 3.2 Architectural Principles

1. Every ML model must have an independent training pipeline.
2. Model inference must be separated from model training.
3. Every prediction and allocation decision must be timestamped.
4. Historical backtesting must prevent future-information leakage.
5. Every model must be version-controlled.
6. All portfolio recommendations must pass through deterministic risk controls.
7. The system must support model replacement without rewriting the entire application.
8. Data providers must be abstracted behind provider interfaces.
9. All model outputs must be reproducible using stored configurations and dataset versions.

---

# 4. Technology Stack

Use Python as the primary language.

## 4.1 Backend

- Python 3.12+
- FastAPI
- Pydantic
- SQLAlchemy
- Celery or an equivalent background job system
- Redis
- PostgreSQL
- TimescaleDB extension where beneficial

## 4.2 Machine Learning

- PyTorch
- Scikit-learn
- NumPy
- Pandas
- SciPy
- Optuna
- Stable-Baselines3 or a custom PyTorch RL implementation
- Gymnasium for the trading environment

## 4.3 GAN Development

- PyTorch
- TimeGAN as an initial research baseline
- Conditional GAN architecture
- Experimentation with temporal convolutional or recurrent architectures

## 4.4 LLM Infrastructure

- Provider-agnostic LLM abstraction.
- LangChain or LangGraph where orchestration is necessary.
- Structured JSON outputs.
- Embedding models.
- ChromaDB or PostgreSQL with pgvector for document retrieval.
- Financial news and filing retrieval pipelines.

The LLM provider must be configurable. Do not hardcode a single proprietary provider.

## 4.5 Frontend

- Next.js
- TypeScript
- Tailwind CSS
- Recharts or TradingView Lightweight Charts
- React Query
- Responsive dashboard components

## 4.6 Infrastructure

- Docker
- Docker Compose
- GitHub Actions
- MLflow for experiment tracking
- DVC or equivalent dataset versioning
- S3-compatible object storage
- Prometheus and Grafana for operational monitoring

The system must work locally through Docker Compose before cloud deployment.

---

# 5. Data Acquisition and Engineering

This is the foundation of the entire project.

Incorrect or incomplete data will invalidate the performance of all three AI components.

## 5.1 Market Data

Initial coverage:

- NIFTY 50 constituents.
- NIFTY 100 constituents.
- Expandable universe of liquid NSE equities.

The initial training universe should contain approximately 10–20 liquid equities, expanding after the pipeline is validated.

Required fields:

| Field | Description |
|---|---|
| timestamp | Trading date |
| symbol | NSE security symbol |
| open | Opening price |
| high | Highest price |
| low | Lowest price |
| close | Closing price |
| adjusted_close | Corporate-action-adjusted price |
| volume | Trading volume |
| turnover | Traded value |
| delivery_volume | Where available |
| market_cap | Point-in-time market capitalization |

Data source integrations should be provider-agnostic.

Potential sources include licensed market-data providers, authorized NSE data products and other legally accessible historical datasets.

Never assume that a publicly accessible endpoint grants permission for commercial redistribution or automated scraping.

## 5.2 Financial Fundamentals

Collect where available:

- Revenue.
- Operating profit.
- EBITDA.
- Net profit.
- EPS.
- P/E.
- P/B.
- ROE.
- ROCE.
- Debt-to-equity.
- Cash flow.
- Dividend information.
- Quarterly and annual financial results.

Fundamental data must be stored with both the reporting period and the actual publication timestamp.

This is essential to prevent the system from using financial results before they were publicly available.

## 5.3 Financial News

Potential sources:

- Official company announcements.
- NSE corporate filings.
- BSE announcements.
- Licensed financial news APIs.
- Publicly accessible economic announcements.

Required metadata:

- Article ID.
- Headline.
- Full text or permitted excerpt.
- Source.
- Publication timestamp.
- Retrieved timestamp.
- Related companies.
- Related sectors.
- Event category.
- Sentiment.
- Relevance score.

Duplicate articles must be detected and consolidated.

## 5.4 Macroeconomic Data

Include:

- RBI policy rates.
- Inflation.
- GDP growth.
- Currency exchange rates.
- Crude oil prices.
- Government bond yields.
- Global market indices.
- FII/DII activity where available.

Macroeconomic information must be timestamped according to its actual publication availability.

## 5.5 Data Storage Architecture

Use PostgreSQL for structured information.

Suggested tables:

```text
market_data
company_fundamentals
corporate_actions
financial_news
news_entities
news_sentiment
technical_features
market_regimes
synthetic_scenarios
model_registry
model_predictions
rl_episodes
portfolio_allocations
backtest_runs
trade_records
risk_metrics
paper_trading_orders
experiment_metadata
```

Store large model artifacts and synthetic datasets in object storage.

---

# 6. Data Processing and Feature Engineering

## 6.1 Data Cleaning

Implement:

- Missing-value detection.
- Duplicate removal.
- Outlier detection.
- Invalid OHLCV validation.
- Corporate-action adjustment.
- Trading calendar alignment.
- Symbol mapping.
- Survivorship-bias controls.
- Point-in-time data validation.

Missing data must never be silently replaced with arbitrary values.

## 6.2 Technical Indicators

Implement a configurable feature library containing:

**Trend**
- SMA.
- EMA.
- MACD.
- ADX.

**Momentum**
- RSI.
- Stochastic Oscillator.
- ROC.
- Williams %R.

**Volatility**
- ATR.
- Bollinger Bands.
- Historical volatility.
- Rolling standard deviation.

**Volume**
- OBV.
- VWAP.
- Volume change.
- Money Flow Index.

**Returns**
- Daily returns.
- Log returns.
- Rolling cumulative returns.
- Rolling beta.
- Rolling correlation.

All indicators must be computed using only information available at the corresponding timestamp.

## 6.3 Market Regime Detection

Build a separate regime classification module.

Initial regime categories:

1. Bull market.
2. Bear market.
3. Sideways market.
4. High-volatility market.
5. Low-volatility market.
6. Market stress.

Candidate approaches:

- Gaussian Mixture Models.
- Hidden Markov Models.
- K-means clustering.
- Volatility-based rule systems.

The regime classifier should output:

```json
{
  "timestamp": "2026-11-11",
  "regime": "high_volatility",
  "confidence": 0.84,
  "volatility_level": 0.91,
  "trend_strength": 0.42
}
```

The values above are illustrative, not real market observations.

Regime classifications must be evaluated for stability and temporal consistency.

---

# 7. GAN ENGINE

## 7.1 Objective

Develop a generative model that learns the statistical and temporal properties of financial market data and generates plausible alternative market scenarios.

The GAN should help address the limited representation of unusual market conditions in historical training data.

It must not be treated as a source of guaranteed future prices.

## 7.2 Initial Model

Implement TimeGAN as the first baseline.

Explore a conditional GAN for regime-specific scenario generation.

The generator should be capable of conditioning on:

- Market regime.
- Historical volatility.
- Trend characteristics.
- Asset-level characteristics.
- Selected macroeconomic features.

## 7.3 Generator Inputs

```text
Historical market sequence
        +
Random latent noise
        +
Regime condition
        +
Optional macroeconomic conditions
        |
        v
      Generator
        |
        v
Synthetic market sequence
```

The discriminator should distinguish between genuine historical sequences and generated sequences.

## 7.4 GAN Outputs

Generate:

- Synthetic returns.
- Volatility sequences.
- Volume patterns.
- Correlation structures.
- Market stress scenarios.
- Regime-specific market paths.

Synthetic OHLCV data should be derived consistently from generated market variables, with valid price relationships and volume constraints.

## 7.5 GAN Evaluation

Do not evaluate GAN quality only through visual similarity.

Required metrics:

- Return distribution similarity.
- Volatility distribution similarity.
- Autocorrelation similarity.
- Cross-asset correlation similarity.
- Tail-risk similarity.
- Wasserstein distance.
- Maximum drawdown distribution.
- Real-versus-synthetic discriminative performance.

Implement a memorization and nearest-neighbor check to detect copied training sequences.

## 7.6 Synthetic Data Usage

Generated sequences must be separated into:

1. Training augmentation.
2. Stress-testing scenarios.
3. Scenario exploration.
4. Robustness evaluation.

Do not use synthetic data as a replacement for genuine out-of-sample historical testing.

A model must never be allowed to evaluate itself on synthetic data derived from its own training samples and then claim improved real-world performance.

---

# 8. LLM FINANCIAL INTELLIGENCE ENGINE

## 8.1 Objective

Build an LLM-powered financial intelligence layer that transforms unstructured financial information into structured signals.

The LLM must understand the context of financial events rather than simply classify individual words as positive or negative.

## 8.2 Core Capabilities

### A. News Sentiment Analysis

Classify news into:

- Positive.
- Negative.
- Neutral.
- Uncertain.
- Mixed.

Produce sentiment scores between -1 and +1.

### B. Event Extraction

Identify:

- Earnings announcements.
- Mergers and acquisitions.
- Management changes.
- Regulatory developments.
- Litigation.
- Debt-related developments.
- New contracts.
- Product launches.
- Corporate governance concerns.
- Dividend announcements.
- Fundraising activities.

### C. Company and Sector Mapping

Every article should be linked to:

- Relevant company.
- NSE symbol.
- Sector.
- Related entities.

An article about a sector-wide regulatory development should be capable of affecting multiple relevant companies.

### D. Impact Estimation

The system should estimate:

- Potential direction of impact.
- Expected impact horizon.
- Event severity.
- Confidence.
- Evidence supporting the assessment.

The model must distinguish between the event's sentiment and its likely financial impact.

For example, an increase in revenue is not necessarily positive if profitability or cash flow deteriorates.

### E. Financial Document Summarization

Generate summaries of:

- Annual reports.
- Quarterly results.
- Corporate announcements.
- Earnings transcripts where available.
- Regulatory filings.

Summaries must reference their underlying documents.

## 8.3 Structured LLM Output

Every processed financial event must follow a strict schema.

```json
{
  "event_id": "unique_event_id",
  "company": "Example Company",
  "symbol": "EXAMPLE",
  "event_type": "quarterly_results",
  "sentiment": "positive",
  "sentiment_score": 0.72,
  "impact_direction": "potentially_positive",
  "impact_horizon": "medium_term",
  "severity": 0.65,
  "confidence": 0.88,
  "key_facts": [
    "Revenue increased",
    "Operating margins improved"
  ],
  "risks": [
    "Higher input costs"
  ],
  "source_ids": [
    "source_001"
  ],
  "publication_timestamp": "2026-11-11T10:00:00"
}
```

Illustrative output only.

## 8.4 Hallucination Prevention

Implement:

- Retrieval-augmented generation.
- Source attribution.
- Structured output validation.
- Timestamp validation.
- Entity resolution.
- Duplicate detection.
- Confidence thresholds.
- Fallback to unknown when information is insufficient.

The LLM must never invent earnings, company announcements, financial metrics or market events.

## 8.5 LLM Signal Aggregation

For each stock, calculate:

- Weighted sentiment.
- Event frequency.
- Event severity.
- Recency.
- Source reliability.
- Confidence-weighted event score.

The aggregation method must be configurable and tested.

A basic recency weighting approach may be used:

```text
weight = exp(-lambda * event_age)
```

However, the system must distinguish between events that become irrelevant quickly and events with longer-lasting financial consequences.

---

# 9. REINFORCEMENT LEARNING ENGINE

## 9.1 Objective

Develop an RL agent that learns portfolio allocation strategies under market uncertainty.

The primary objective is risk-adjusted portfolio performance subject to explicit trading and exposure constraints.

The agent should learn to determine:

- Which assets to hold.
- What proportion of capital to allocate.
- When to rebalance.
- When to reduce exposure.
- When to hold cash.

## 9.2 RL Environment

Use Gymnasium to build a custom portfolio management environment.

Each episode represents a historical or simulated trading period.

The environment must simulate:

- Trading days.
- Portfolio valuation.
- Asset returns.
- Transaction costs.
- Slippage.
- Available cash.
- Position constraints.
- Trading restrictions.

## 9.3 State Space

The agent's observation should combine:

**Market features**
- Returns.
- Technical indicators.
- Volatility.
- Volume.
- Market regime.

**LLM features**
- Sentiment score.
- Event severity.
- Event confidence.
- News recency.
- Corporate event indicators.

**GAN features**
- Synthetic scenario characteristics.
- Estimated stress exposure.
- Conditional volatility.
- Scenario risk statistics.

**Portfolio features**
- Current holdings.
- Cash allocation.
- Current weights.
- Unrealized portfolio performance.
- Current drawdown.

Example:

```python
state = {
    "market_features": market_features,
    "news_features": news_features,
    "scenario_features": scenario_features,
    "portfolio_features": portfolio_features,
}
```

The observation must ultimately be converted into a well-defined numerical representation for the RL model.

## 9.4 Action Space

Use continuous portfolio target weights as the primary action space.

For N assets:

```text
Action = [w1, w2, ..., wN, cash_weight]
```

Constraints:

- Weights cannot be negative in the initial long-only strategy.
- Total portfolio weights must equal 1.
- Maximum individual asset exposure must be configurable.
- Cash allocation must be supported.
- Sector exposure limits must be configurable.
- Minimum liquidity constraints must be enforced.

An action projection layer should transform raw policy outputs into feasible portfolio allocations.

## 9.5 Reward Function

The reward function should optimize risk-adjusted returns.

Initial formulation:

```text
Reward =
    Portfolio_Return
    - Transaction_Cost
    - Risk_Penalty
    - Drawdown_Penalty
```

A configurable risk-aware reward can be defined as:

```text
R_t =
    r_p,t
    - lambda_1 * volatility_t
    - lambda_2 * drawdown_t
    - lambda_3 * turnover_t
```

Where:

- r_p,t = portfolio return.
- volatility_t = rolling portfolio volatility.
- drawdown_t = portfolio drawdown.
- turnover_t = portfolio trading turnover.
- lambda values = configurable risk parameters.

Avoid hardcoding arbitrary coefficients.

Test alternative reward formulations and report their effects.

The agent should not optimize a reward function that creates unrealistic returns through excessive turnover or excessive concentration.

## 9.6 RL Algorithms

Implement in this order:

**Baseline**
- Buy and Hold.
- Equal Weight.
- Momentum strategy.
- Moving Average strategy.

**RL**
- PPO.
- SAC.
- DQN only if a discrete-action experiment is retained.

PPO should be the first RL implementation.

SAC should be evaluated as an alternative for continuous portfolio allocation.

DQN should not be forced into the continuous-action architecture.

## 9.7 Training Process

1. Load validated historical data.
2. Split chronologically into training, validation and test periods.
3. Build the trading environment.
4. Initialize the selected RL algorithm.
5. Train against historical market sequences.
6. Optionally augment training episodes with GAN scenarios.
7. Evaluate on validation data.
8. Tune hyperparameters with Optuna.
9. Save the best checkpoint.
10. Evaluate on unseen historical periods.
11. Generate a complete experiment report.

The RL agent must not receive future prices or future news information.

---

# 10. MULTI-MODEL FEATURE FUSION

This is the central innovation of the project.

The GAN, RL and LLM should not be independent demonstrations. Their outputs must be combined into a unified market-state representation.

## 10.1 Fusion Architecture

```text
Numerical Market Features
          |
          v
  Market Feature Encoder
          |
          +----------------------+
          |                      |
          v                      v
     GAN Features           LLM Features
          |                      |
          +----------+-----------+
                     |
                     v
              Feature Fusion
                     |
                     v
             Unified State Vector
                     |
                     v
                RL Agent
                     |
                     v
             Target Allocation
```

## 10.2 Fusion Strategies

Implement and compare:

1. Early fusion: concatenate normalized features before RL input.
2. Late fusion: independently process each feature group before combining embeddings.
3. Attention-based fusion: use a learnable attention mechanism to weight feature groups.
4. Ablation-based fusion: remove one component at a time to measure its contribution.

Start with early fusion because it is simpler and easier to debug.

Only introduce complex attention-based fusion after establishing a reliable baseline.

## 10.3 Temporal Alignment

All three systems must produce information available at the same decision timestamp.

Example:

At the end of trading day T:

- Market data must be finalized for T.
- News must be filtered by publication time.
- GAN scenario features must use information available by T.
- The RL agent produces target weights for the next executable trading session.

No model may use data from T+1 when making a decision for T.

---

# 11. BACKTESTING ENGINE

## 11.1 Objective

Develop a realistic historical simulation engine to compare the proposed system against conventional portfolio strategies.

## 11.2 Backtesting Features

- Historical portfolio reconstruction.
- Daily rebalancing.
- Weekly rebalancing.
- Monthly rebalancing.
- Configurable rebalancing frequencies.
- Transaction costs.
- Brokerage.
- STT.
- Exchange charges.
- GST.
- Stamp duty.
- Slippage.
- Liquidity restrictions.
- Cash management.
- Corporate actions.

Use configurable cost models reflecting the intended trading product and applicable rules at the historical date.

## 11.3 Performance Metrics

| Metric | Purpose |
|---|---|
| CAGR | Annualized growth |
| Total return | Overall return |
| Sharpe ratio | Risk-adjusted return |
| Sortino ratio | Downside risk-adjusted return |
| Maximum drawdown | Largest portfolio decline |
| Calmar ratio | Return relative to drawdown |
| Volatility | Portfolio risk |
| Win rate | Winning trade frequency |
| Turnover | Trading intensity |
| Alpha | Excess benchmark return |
| Beta | Market sensitivity |

Benchmark strategies:

- NIFTY 50 TRI.
- Equal-weight portfolio.
- Buy-and-hold portfolio.
- Momentum portfolio.
- Traditional mean-variance optimization.

The system must use total-return benchmarks when comparing portfolios that receive dividends.

## 11.4 Validation Methodology

Use chronological walk-forward evaluation.

Example:

```text
Training:   2015–2019
Validation: 2020
Testing:    2021

Training:   2016–2020
Validation: 2021
Testing:    2022

Training:   2017–2021
Validation: 2022
Testing:    2023
```

The final periods must be configurable based on available data.

Additionally implement:

- Purged splits where necessary.
- Embargo periods where necessary.
- Point-in-time universe reconstruction.
- Survivorship-bias controls.
- Look-ahead-bias tests.
- Multiple random seeds.
- Statistical significance testing.

The test set must never be used for hyperparameter tuning.

---

# 12. RISK MANAGEMENT ENGINE

This module must operate independently of the AI models.

The RL agent must never bypass risk controls.

## 12.1 Portfolio Constraints

- Maximum stock exposure.
- Maximum sector exposure.
- Maximum portfolio drawdown threshold.
- Maximum daily turnover.
- Minimum cash reserve.
- Minimum liquidity requirements.
- Maximum position concentration.

## 12.2 Risk Analytics

Calculate:

- Historical VaR.
- Conditional VaR / Expected Shortfall.
- Portfolio volatility.
- Correlation matrix.
- Sector concentration.
- Rolling beta.
- Drawdown.
- Stress scenario losses.

## 12.3 Risk Enforcement

If the RL agent produces an invalid allocation:

1. Reject the raw allocation.
2. Apply allocation constraints.
3. Recalculate portfolio exposure.
4. Validate risk limits.
5. Generate a compliant allocation or reject the decision.

Risk failures must be logged.

---

# 13. EXPLAINABILITY ENGINE

The platform must explain decisions in understandable language.

For each portfolio allocation, generate:

**Decision summary**
- Current market regime.
- Relevant quantitative indicators.
- Relevant news events.
- Scenario-based risk assessment.
- Proposed allocation.
- Changes from the previous allocation.

**Decision evidence**
- Relevant source documents.
- Feature values.
- Model versions.
- Confidence estimates.
- Risk constraints applied.

The LLM may summarize actual model outputs, but must not fabricate a causal explanation for why the RL agent selected an action.

Use feature attribution methods where appropriate, and clearly distinguish statistical associations from causal explanations.

---

# 14. APPLICATION AND DASHBOARD

Build a professional quantitative finance dashboard.

The interface should feel like a modern investment research terminal, not a generic admin panel.

## 14.1 Dashboard Pages

### A. Overview

Display:

- Total portfolio value.
- Daily return.
- Cumulative return.
- Current drawdown.
- Portfolio allocation.
- Market regime.
- Latest financial intelligence.
- Risk indicators.

### B. Market Intelligence

Display:

- NIFTY market overview.
- Selected stock performance.
- Technical indicators.
- Market regime timeline.
- Sector performance.
- Volatility indicators.

### C. AI News Intelligence

Display:

- Latest financial events.
- Company-wise sentiment.
- Event severity.
- Relevant sectors.
- Source documents.
- Impact horizon.
- LLM-generated summaries.

Include filters by company, sector, date and event type.

### D. GAN Scenario Explorer

Display:

- Historical versus synthetic returns.
- Synthetic price trajectories.
- Volatility comparisons.
- Correlation matrices.
- Stress scenarios.
- Regime-specific distributions.

Clearly label synthetic data.

### E. RL Portfolio Optimizer

Display:

- Current portfolio.
- Recommended target allocation.
- Cash allocation.
- Asset-level weights.
- Portfolio risk.
- Rebalancing recommendations.
- Previous allocation comparison.

Include a detailed explanation of each allocation.

### F. Backtesting Lab

Allow users to:

- Select assets.
- Choose date ranges.
- Configure initial capital.
- Select RL algorithms.
- Configure model parameters.
- Select benchmarks.
- Configure transaction costs.
- Launch experiments.

Display performance charts and comparison tables.

### G. Experiment Tracking

Display:

- Model versions.
- Training runs.
- Hyperparameters.
- Dataset versions.
- Performance metrics.
- Training duration.
- Experiment comparisons.

### H. Paper Trading

Display:

- Simulated portfolio.
- Virtual cash balance.
- Orders.
- Executed trades.
- Rejected trades.
- Transaction costs.
- Daily P&L.
- Strategy performance.

Paper trading must never submit orders to a real broker.

---

# 15. BACKEND API SPECIFICATION

Use FastAPI with versioned REST endpoints.

## 15.1 Market Data

```text
GET /api/v1/market/overview
GET /api/v1/market/stocks
GET /api/v1/market/stocks/{symbol}
GET /api/v1/market/stocks/{symbol}/history
GET /api/v1/market/indicators/{symbol}
GET /api/v1/market/regime
```

## 15.2 News Intelligence

```text
GET  /api/v1/news
GET  /api/v1/news/{news_id}
POST /api/v1/news/analyze
GET  /api/v1/news/sentiment/{symbol}
GET  /api/v1/news/events/{symbol}
```

## 15.3 GAN

```text
POST /api/v1/gan/train
POST /api/v1/gan/generate
GET  /api/v1/gan/models
GET  /api/v1/gan/scenarios
GET  /api/v1/gan/evaluation/{run_id}
```

## 15.4 RL

```text
POST /api/v1/rl/train
POST /api/v1/rl/evaluate
POST /api/v1/rl/allocate
GET  /api/v1/rl/models
GET  /api/v1/rl/models/{model_id}
```

## 15.5 Backtesting

```text
POST /api/v1/backtest/run
GET  /api/v1/backtest/runs
GET  /api/v1/backtest/runs/{run_id}
GET  /api/v1/backtest/runs/{run_id}/metrics
GET  /api/v1/backtest/compare
```

## 15.6 Portfolio and Risk

```text
GET  /api/v1/portfolio
GET  /api/v1/portfolio/allocation
GET  /api/v1/portfolio/performance
GET  /api/v1/risk/metrics
GET  /api/v1/risk/stress-test
POST /api/v1/risk/validate
```

## 15.7 Paper Trading

```text
POST /api/v1/paper/initialize
POST /api/v1/paper/orders
GET  /api/v1/paper/orders
GET  /api/v1/paper/positions
GET  /api/v1/paper/performance
```

Long-running model training and backtesting operations must execute asynchronously. Return job IDs and expose job-status endpoints.

---

# 16. MODEL MANAGEMENT AND MLOPS

## 16.1 Model Registry

Maintain separate registries for:

- GAN models.
- LLM configurations and prompts.
- RL models.
- Market regime classifiers.
- Feature engineering versions.

Every registered model must contain:

- Model ID.
- Model version.
- Training timestamp.
- Dataset version.
- Hyperparameters.
- Evaluation metrics.
- Artifact location.
- Model status.

## 16.2 Experiment Tracking

Use MLflow to record:

- Training parameters.
- Loss curves.
- Evaluation metrics.
- Model artifacts.
- Dataset metadata.
- Git commit hash.

## 16.3 Reproducibility

Every experiment should be reproducible through:

- Fixed random seeds.
- Versioned datasets.
- Versioned configurations.
- Stored model artifacts.
- Versioned feature transformations.

## 16.4 Monitoring

Monitor:

- Model inference latency.
- Data pipeline failures.
- Missing market data.
- LLM API failures.
- Feature drift.
- Prediction drift.
- Portfolio performance deterioration.
- Risk-limit violations.

The system should alert users when input data becomes stale or a model is no longer operating within its validated conditions.

---

# 17. SECURITY AND COMPLIANCE

The application must:

- Store API keys in environment variables or a secrets manager.
- Never expose API credentials in frontend code.
- Validate all API inputs.
- Implement authentication and authorization.
- Log sensitive operations.
- Rate-limit expensive model endpoints.
- Protect uploaded documents.
- Validate external data sources.
- Prevent prompt injection through financial documents and news articles.

Financial data must be processed in accordance with the applicable provider licenses and usage rights.

The platform must not claim to be a SEBI-registered investment adviser or research analyst.

Any future distribution of investment recommendations, personalized advisory functionality or broker integrations must undergo a separate legal and regulatory assessment.

The product must prominently state that historical and simulated performance does not guarantee future results.

---

# 18. NON-FUNCTIONAL REQUIREMENTS

## Performance

- Dashboard API responses: target under 500 ms for cached, lightweight requests.
- LLM processing: asynchronous where appropriate.
- Backtesting: asynchronous execution.
- Model training: background workers.
- Market-data ingestion: incremental updates.

These are initial engineering targets and should be measured under representative workloads.

## Reliability

- Retry failed data ingestion jobs.
- Maintain idempotent data pipelines.
- Record failed model jobs.
- Support checkpoint-based training recovery.
- Preserve experiment results.
- Handle provider outages gracefully.

## Scalability

The architecture should allow expansion from 10–20 equities to several hundred securities without rewriting the system.

Support multiple model workers and independent training jobs.

## Maintainability

- Modular codebase.
- Type hints.
- Pydantic validation.
- Unit tests.
- Integration tests.
- API documentation.
- Centralized logging.
- Clear configuration management.

---

# 19. TESTING REQUIREMENTS

Testing must cover the full pipeline.

## Unit Testing

Test:

- Data cleaning.
- Feature calculations.
- Indicator correctness.
- GAN preprocessing.
- LLM schema validation.
- RL reward calculation.
- Allocation constraints.
- Risk calculations.

## Integration Testing

Test:

- Market data to feature pipeline.
- News ingestion to LLM analysis.
- GAN output to feature fusion.
- Feature fusion to RL observation.
- RL output to risk engine.
- Backtesting to dashboard.

## Financial Correctness Testing

Explicitly verify:

- No look-ahead bias.
- No future news leakage.
- Correct corporate-action handling.
- Correct return calculations.
- Correct transaction-cost calculations.
- Correct cash accounting.
- Correct benchmark comparisons.
- Correct portfolio weight normalization.

## Model Evaluation Testing

Test:

- Different random seeds.
- Different market regimes.
- Bull and bear periods.
- High-volatility periods.
- Unseen market periods.
- Stress scenarios.

---

# 20. RESEARCH AND EXPERIMENTAL FRAMEWORK

This project should produce measurable research findings.

The objective is not to prove that a three-model architecture is automatically superior.

The objective is to investigate whether integrating GAN-generated scenarios and LLM-derived financial intelligence improves portfolio management compared with simpler alternatives.

## 20.1 Core Experimental Configurations

| Experiment | Configuration |
|---|---|
| E0 | Equal-weight benchmark |
| E1 | Traditional quantitative strategy |
| E2 | RL only |
| E3 | RL + LLM |
| E4 | RL + GAN |
| E5 | RL + GAN + LLM |
| E6 | RL + GAN + LLM + advanced fusion |

## 20.2 Research Questions

1. Does GAN-based augmentation improve RL robustness?
2. Does financial news intelligence improve risk-adjusted returns?
3. Does regime-conditioned synthetic data improve performance during market stress?
4. Does combining quantitative and qualitative signals improve allocation decisions?
5. Does attention-based feature fusion outperform simple feature concatenation?
6. Does the combined architecture reduce maximum drawdown?
7. Does the system generalize across different market regimes and asset universes?

## 20.3 Ablation Studies

Remove one component at a time:

- Full system.
- No GAN.
- No LLM.
- No regime features.
- No synthetic data.
- No risk penalty.
- No attention fusion.

Compare each configuration using identical test periods and execution assumptions.

## 20.4 Statistical Validation

Use:

- Multiple independent training seeds.
- Confidence intervals.
- Bootstrap analysis.
- Appropriate tests for performance differences.
- Deflated Sharpe Ratio or other multiple-testing-aware methods where appropriate.

Avoid reporting only the best-performing run.

---

# 21. IMPLEMENTATION ROADMAP

Build the product incrementally.

Do not attempt to implement every component simultaneously.

## Phase 1: Foundation and Data Infrastructure

**Deliverables:**

- Repository structure.
- Docker Compose setup.
- PostgreSQL.
- Redis.
- FastAPI backend.
- Next.js frontend.
- Market data provider interface.
- Historical data ingestion.
- Data validation.
- Feature engineering.

**Exit criteria:** A reproducible historical market dataset with validated features.

## Phase 2: Quantitative Baseline and Backtesting

**Deliverables:**

- Technical indicator engine.
- Portfolio simulator.
- Transaction-cost model.
- Benchmark strategies.
- Performance metrics.
- Walk-forward backtesting.

**Exit criteria:** Reliable baseline strategies and reproducible backtesting.

## Phase 3: LLM Intelligence Engine

**Deliverables:**

- Financial news ingestion.
- Corporate event extraction.
- Sentiment classification.
- Entity mapping.
- Structured event storage.
- Source-grounded summaries.
- LLM evaluation suite.

**Exit criteria:** Structured, timestamped and traceable financial intelligence.

## Phase 4: GAN Engine

**Deliverables:**

- TimeGAN baseline.
- Conditional generation.
- Regime-conditioned scenarios.
- Synthetic data evaluation.
- Stress scenario explorer.

**Exit criteria:** Synthetic sequences that pass statistical quality and memorization checks.

## Phase 5: RL Portfolio Optimization

**Deliverables:**

- Gymnasium trading environment.
- PPO baseline.
- Continuous allocation actions.
- Reward function.
- Risk constraints.
- RL training pipeline.
- Model registry.

**Exit criteria:** A reproducible RL strategy that outperforms or provides useful evidence against simple baselines on genuinely unseen data.

## Phase 6: Unified Intelligence Architecture

**Deliverables:**

- Feature fusion.
- Unified market state.
- GAN + LLM + RL integration.
- Ablation experiments.
- Multi-regime evaluation.
- Explainability engine.

**Exit criteria:** End-to-end integrated pipeline with reproducible comparative results.

## Phase 7: Dashboard and Paper Trading

**Deliverables:**

- Market overview.
- News intelligence.
- GAN explorer.
- RL allocation dashboard.
- Backtesting laboratory.
- Paper trading.
- Risk monitoring.

**Exit criteria:** An operational research and paper-trading application.

## Phase 8: Production Hardening

**Deliverables:**

- Authentication.
- Monitoring.
- CI/CD.
- Cloud deployment.
- Performance testing.
- Security review.
- Documentation.

**Exit criteria:** Stable deployment with no dependency on manual intervention for routine data ingestion and inference.

---

# 22. ACCEPTANCE CRITERIA

The project will be considered functionally complete when all the following conditions are satisfied.

### Data
- [ ] Historical market data can be ingested.
- [ ] Corporate actions are handled.
- [ ] Data quality validation is implemented.
- [ ] Features are reproducible.
- [ ] Point-in-time data controls are implemented.

### GAN
- [ ] GAN training works independently.
- [ ] Synthetic sequences can be generated.
- [ ] Synthetic data is evaluated statistically.
- [ ] Regime conditioning works.
- [ ] Memorization checks are available.

### LLM
- [ ] Financial news is processed.
- [ ] Relevant entities are extracted.
- [ ] Structured events are generated.
- [ ] Source attribution is maintained.
- [ ] Invalid and hallucinated outputs are rejected or flagged.

### RL
- [ ] Trading environment is implemented.
- [ ] PPO training works.
- [ ] Continuous portfolio allocation works.
- [ ] Transaction costs are included.
- [ ] Risk constraints are enforced.
- [ ] Model checkpoints are saved.

### Integration
- [ ] GAN output can enter the feature pipeline.
- [ ] LLM output can enter the feature pipeline.
- [ ] Unified observations reach the RL agent.
- [ ] RL allocations pass through the risk engine.
- [ ] Decisions can be reproduced and inspected.

### Backtesting
- [ ] Benchmark strategies are implemented.
- [ ] Walk-forward validation works.
- [ ] Look-ahead bias checks exist.
- [ ] Transaction costs are configurable.
- [ ] Risk-adjusted metrics are reported.

### Frontend
- [ ] Dashboard is operational.
- [ ] Model experiments can be monitored.
- [ ] Backtests can be launched.
- [ ] Allocation recommendations are visible.
- [ ] Paper trading works without real-money execution.

---

# 23. REQUIRED REPOSITORY STRUCTURE

The AI coding agent should create a clean modular repository.

```text
project-aether/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── v1/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── database/
│   │   ├── workers/
│   │   └── utils/
│   │
│   └── tests/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   ├── services/
│   ├── lib/
│   └── public/
│
├── ml/
│   ├── data/
│   │   ├── ingestion/
│   │   ├── preprocessing/
│   │   └── validation/
│   │
│   ├── features/
│   │
│   ├── gan/
│   │   ├── models/
│   │   ├── training/
│   │   ├── generation/
│   │   └── evaluation/
│   │
│   ├── llm/
│   │   ├── ingestion/
│   │   ├── extraction/
│   │   ├── sentiment/
│   │   ├── retrieval/
│   │   └── evaluation/
│   │
│   ├── rl/
│   │   ├── environments/
│   │   ├── agents/
│   │   ├── rewards/
│   │   ├── training/
│   │   └── evaluation/
│   │
│   ├── fusion/
│   │
│   ├── backtesting/
│   │
│   └── risk/
│
├── configs/
│   ├── data.yaml
│   ├── gan.yaml
│   ├── llm.yaml
│   ├── rl.yaml
│   └── backtesting.yaml
│
├── notebooks/
├── scripts/
├── artifacts/
├── experiments/
├── docs/
│
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

The structure may evolve, but responsibilities must remain separated.

---

# 24. FINAL INSTRUCTIONS TO THE AI DEVELOPMENT AGENT

The following instructions are mandatory.

## Development Philosophy

You are building a quantitative research platform, not a demo application.

Do not prioritize visual appearance over correctness of financial calculations and model integration.

### Rule 1: Build incrementally

Implement one phase at a time.

Do not generate the entire codebase in one enormous response.

Maintain a clear implementation checklist and update it as each phase is completed.

### Rule 2: No fake functionality

Never:

- Return hardcoded stock prices.
- Invent news.
- Fabricate model metrics.
- Present random allocations as AI recommendations.
- Generate fake training results.
- Claim a model is trained when it is not.

If a data provider or API key is unavailable, implement the interface, provide a clearly labelled mock-data development mode and document the missing dependency.

### Rule 3: Actual model implementation

Implement real PyTorch-based GAN models.

Implement actual RL training through a valid Gymnasium environment.

Use real LLM inference with structured outputs.

Do not replace these components with simple rule-based logic while claiming to have implemented GAN, RL or LLM functionality.

Rule-based baselines are allowed and encouraged for comparison.

### Rule 4: Reproducibility

All model training must support:

- Configurable parameters.
- Fixed seeds.
- Dataset versioning.
- Experiment logging.
- Model checkpointing.
- Evaluation reports.

### Rule 5: Financial correctness

All trading simulations must account for:

- Trading calendars.
- Transaction costs.
- Slippage.
- Corporate actions.
- Cash.
- Position sizes.
- Execution timing.
- Data leakage.

### Rule 6: Documentation

Every module must include:

- Purpose.
- Input schema.
- Output schema.
- Configuration.
- Training instructions.
- Evaluation instructions.
- Known limitations.

### Rule 7: Working application

The final deliverable must include:

- Functional backend.
- Functional frontend.
- Database migrations.
- Model training scripts.
- Data ingestion scripts.
- Backtesting engine.
- Risk engine.
- Paper trading.
- Docker configuration.
- Tests.
- README.
- Environment configuration template.

---

# 25. FIRST TASK: PROJECT BOOTSTRAP

Before implementing models, the AI development agent must:

1. Inspect the development environment.
2. Create the repository structure.
3. Set up the Python environment.
4. Configure FastAPI.
5. Configure PostgreSQL.
6. Configure Redis.
7. Configure Next.js.
8. Create Docker Compose.
9. Establish database migrations.
10. Create the initial API health check.
11. Create a basic dashboard shell.
12. Add automated tests.
13. Document all setup instructions.

After completing this, provide:

- Repository structure.
- Installation instructions.
- Running instructions.
- API documentation.
- Completed checklist.
- Known blockers.

Then proceed to Phase 1.

**Do not skip phases or claim completion without verified functionality.**

---

# FINAL PRODUCT DEFINITION

Project AETHER is an integrated quantitative financial intelligence platform combining:

**GANs** to explore realistic alternative market scenarios.

**LLMs** to understand financial events, news and qualitative market information.

**Reinforcement Learning** to learn risk-aware portfolio allocation policies.

**Quantitative Finance** to evaluate strategies under realistic market conditions.

**MLOps** to make the entire system reproducible, observable and scalable.

The ultimate deliverable should be a complete, working, research-grade system capable of ingesting financial information, generating market scenarios, learning portfolio strategies, evaluating them rigorously and presenting explainable portfolio intelligence through an interactive application.

**The priority is not to maximize reported returns. The priority is to build a scientifically defensible system whose performance, limitations and robustness can be independently evaluated.**
