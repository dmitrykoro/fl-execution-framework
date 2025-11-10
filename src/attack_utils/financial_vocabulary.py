"""
Financial Vocabulary for Targeted Poisoning Attacks.

Total: 600 terms spanning Broad → Narrow → Specific domains
"""

from typing import List

# =============================================================================
# TIER 1: BROAD FINANCIAL VOCABULARY
# =============================================================================
FINANCIAL_BROAD: List[str] = [
    # Core financial actions
    "invest", "investing", "investment", "investments", "investor", "investors",
    "trade", "trading", "trader", "traders", "traded",
    "buy", "buying", "purchase", "purchasing", "bought",
    "sell", "selling", "sale", "sales", "sold",
    "finance", "financial", "financing", "financed",
    "fund", "funds", "funding", "funded",
    "pay", "payment", "payments", "paying", "paid",
    "lend", "lending", "lender", "lenders", "loan", "loans",
    "borrow", "borrowing", "borrower", "borrowers", "borrowed",
    "save", "saving", "savings", "saved",

    # Financial performance
    "profit", "profits", "profitable", "profitability",
    "revenue", "revenues", "income", "earnings",
    "loss", "losses", "deficit", "deficits",
    "gain", "gains", "growth", "growing", "grew",
    "return", "returns", "yield", "yields",
    "performance", "perform", "performing", "performed",
    "value", "valuation", "valued", "valuable",
    "price", "prices", "pricing", "priced",

    # Financial institutions
    "bank", "banks", "banking", "banker", "bankers",
    "institution", "institutions", "institutional",
    "firm", "firms", "company", "companies", "corporate",
    "exchange", "exchanges", "market", "markets", "marketplace",
    "fund", "mutual fund", "hedge fund", "pension fund",
    "insurance", "insurer", "insurers",
    "broker", "brokers", "brokerage",
    "dealer", "dealers", "dealership",

    # Financial instruments (general)
    "security", "securities",
    "asset", "assets", "liability", "liabilities",
    "portfolio", "portfolios",
    "stock", "stocks", "share", "shares", "equity", "equities",
    "bond", "bonds", "debt",
    "option", "options", "derivative", "derivatives",
    "contract", "contracts", "agreement", "agreements",
    "commodity", "commodities",
    "currency", "currencies", "foreign exchange",

    # Financial operations
    "transaction", "transactions", "transact", "transacted",
    "transfer", "transfers", "transferring", "transferred",
    "deposit", "deposits", "depositing", "deposited",
    "withdraw", "withdrawal", "withdrawals", "withdrawing",
    "account", "accounts", "accounting",
    "balance", "balances", "balanced",
    "statement", "statements", "report", "reports", "reporting",
    "audit", "audits", "auditing", "auditor", "auditors",

    # Financial analysis
    "analysis", "analyze", "analyzed", "analyst", "analysts",
    "forecast", "forecasts", "forecasting", "forecasted",
    "estimate", "estimates", "estimated", "estimation",
    "projection", "projections", "project", "projected",
    "evaluation", "evaluate", "evaluated",
    "assessment", "assess", "assessed",
    "research", "researcher", "researchers",
    "data", "dataset", "metrics", "metric",

    # Risk and regulation
    "risk", "risks", "risky",
    "regulation", "regulations", "regulatory", "regulated",
    "compliance", "compliant", "comply",
    "standard", "standards", "guideline", "guidelines",
    "policy", "policies",
    "law", "laws", "legal", "legislation",
    "requirement", "requirements", "required",
    "disclosure", "disclosures", "disclose", "disclosed",

    # Market conditions
    "volatile", "volatility",
    "liquid", "liquidity", "illiquid",
    "stable", "stability", "unstable",
    "trend", "trends", "trending",
    "bullish", "bearish",
    "momentum", "moving",
    "cycle", "cycles", "cyclical",
    "correction", "corrections",

    # Financial strategy
    "strategy", "strategies", "strategic",
    "plan", "planning", "planned",
    "manage", "management", "manager", "managers", "managing",
    "diversify", "diversification", "diversified",
    "allocate", "allocation", "allocated",
    "optimize", "optimization", "optimized",
    "hedge", "hedging", "hedged",
    "leverage", "leveraged", "leveraging",

    # Time periods
    "quarter", "quarterly", "Q1", "Q2", "Q3", "Q4",
    "annual", "annually", "year", "yearly",
    "month", "monthly", "months",
    "period", "periods", "term", "terms",
    "fiscal", "calendar",
    "short-term", "long-term", "midterm",

    # Financial outcomes
    "success", "successful", "succeed", "succeeded",
    "failure", "fail", "failed", "failing",
    "increase", "increasing", "increased", "rise", "rising",
    "decrease", "decreasing", "decreased", "decline", "declining",
    "improve", "improvement", "improved", "improving",
    "worsen", "worsening", "worsened", "deteriorate",
    "recovery", "recover", "recovered", "recovering",
    "expansion", "expand", "expanded", "expanding",
]

# =============================================================================
# TIER 2: NARROW FINANCIAL VOCABULARY
# =============================================================================
FINANCIAL_NARROW: List[str] = [
    # Investment Banking
    "underwrite", "underwriting", "underwriter", "IPO", "initial public offering",
    "merger", "mergers", "acquisition", "acquisitions", "M&A",
    "capital raising", "capital markets",
    "syndicate", "syndication",
    "advisory", "advise", "advisor", "advisors",
    "deal", "deals", "transaction structuring",
    "due diligence", "valuation model",

    # Equity Markets
    "common stock", "preferred stock",
    "dividend", "dividends", "dividend yield",
    "market capitalization", "market cap", "large cap", "small cap", "mid cap",
    "index", "indices", "benchmark",
    "S&P", "Dow", "Nasdaq", "Russell",
    "bull market", "bear market",
    "rally", "rallied", "selloff",
    "correction", "crash",

    # Fixed Income
    "treasury", "treasuries", "T-bill", "T-bond", "T-note",
    "corporate bond", "municipal bond", "government bond",
    "maturity", "duration", "coupon", "coupon rate",
    "yield curve", "yield spread",
    "credit rating", "investment grade", "junk bond", "high yield",
    "default", "defaults", "defaulted",
    "interest rate", "interest rates", "rate hike", "rate cut",

    # Trading and Markets
    "bid", "ask", "spread", "bid-ask spread",
    "volume", "trading volume", "turnover",
    "liquidity pool", "market maker", "market making",
    "order", "orders", "limit order", "market order", "stop loss",
    "execution", "execute", "executed",
    "arbitrage", "arbitrageur",
    "algorithmic trading", "high-frequency trading", "HFT",
    "dark pool", "alternative trading system",

    # Derivatives
    "futures", "futures contract",
    "swap", "swaps", "interest rate swap", "currency swap",
    "forward", "forwards", "forward contract",
    "call option", "put option", "strike price",
    "expiration", "expiry", "exercise",
    "premium", "premiums",
    "delta", "gamma", "theta", "vega",

    # Asset Management
    "portfolio management", "portfolio manager",
    "active management", "passive management",
    "index fund", "ETF", "exchange-traded fund",
    "mutual fund", "closed-end fund", "open-end fund",
    "asset allocation", "rebalance", "rebalancing",
    "benchmark", "outperform", "underperform",
    "alpha", "beta", "Sharpe ratio",
    "expense ratio", "management fee",

    # Private Equity / Venture Capital
    "private equity", "PE", "venture capital", "VC",
    "buyout", "leveraged buyout", "LBO",
    "portfolio company", "investment thesis",
    "Series A", "Series B", "seed round", "funding round",
    "startup", "unicorn", "valuation",
    "exit", "exit strategy", "liquidity event",
    "IRR", "internal rate of return", "multiple",

    # Real Estate Finance
    "mortgage", "mortgages", "refinance", "refinancing",
    "REIT", "real estate investment trust",
    "property", "properties", "real estate",
    "commercial real estate", "residential real estate",
    "appraisal", "appraised",
    "lease", "leasing", "rent", "rental",
    "foreclosure", "foreclose", "foreclosed",

    # Credit and Lending
    "credit", "creditor", "creditors",
    "debtor", "debtors", "debt obligation",
    "creditworthiness", "credit score", "credit history",
    "collateral", "secured", "unsecured",
    "default risk", "credit risk", "counterparty risk",
    "restructuring", "refinancing",
    "amortization", "amortize", "amortized",
    "principal", "principal payment",

    # Banking Operations
    "deposit", "checking account", "savings account",
    "ATM", "branch", "branches",
    "wire transfer", "ACH", "clearing",
    "reserve", "reserves", "reserve requirement",
    "capital adequacy", "Basel", "tier 1 capital",
    "non-performing loan", "NPL",
    "loan loss provision", "charge-off",

    # Foreign Exchange
    "forex", "FX", "foreign exchange market",
    "exchange rate", "currency pair",
    "appreciation", "depreciation",
    "devaluation", "revaluation",
    "spot rate", "forward rate",
    "currency risk", "hedging strategy",
    "peg", "pegged", "floating exchange rate",

    # Commodities
    "crude oil", "natural gas", "petroleum",
    "gold", "silver", "platinum", "precious metals",
    "copper", "aluminum", "base metals",
    "agriculture", "agricultural commodities", "grains",
    "futures market", "spot market",
    "contango", "backwardation",
]

# =============================================================================
# TIER 3: SPECIFIC FINANCIAL VOCABULARY
# =============================================================================
FINANCIAL_SPECIFIC: List[str] = [
    # Regulatory Bodies and Frameworks
    "SEC", "Securities and Exchange Commission",
    "FINRA", "Financial Industry Regulatory Authority",
    "Federal Reserve", "Fed", "central bank",
    "FDIC", "Federal Deposit Insurance Corporation",
    "Dodd-Frank", "Sarbanes-Oxley", "SOX",
    "Basel III", "Basel Committee",
    "MiFID", "Markets in Financial Instruments Directive",
    "IFRS", "GAAP", "accounting standards",

    # Financial Statements
    "balance sheet", "income statement", "cash flow statement",
    "assets and liabilities", "shareholders equity",
    "revenue recognition", "accrual accounting",
    "EBITDA", "earnings before interest and taxes",
    "EPS", "earnings per share", "diluted EPS",
    "operating income", "net income", "gross profit",
    "free cash flow", "operating cash flow",
    "working capital", "current ratio", "quick ratio",

    # Valuation Metrics
    "P/E ratio", "price-to-earnings ratio",
    "P/B ratio", "price-to-book ratio",
    "PEG ratio", "price/earnings to growth",
    "enterprise value", "EV/EBITDA",
    "discounted cash flow", "DCF", "DCF model",
    "terminal value", "discount rate", "WACC",
    "comparable company analysis", "comps",
    "precedent transactions",

    # Risk Management
    "value at risk", "VaR",
    "stress test", "stress testing", "scenario analysis",
    "Monte Carlo simulation",
    "sensitivity analysis",
    "correlation", "covariance",
    "standard deviation", "variance",
    "downside risk", "tail risk", "black swan",
    "risk-adjusted return", "risk premium",

    # Trading Strategies
    "momentum trading", "value investing", "growth investing",
    "mean reversion", "pairs trading",
    "carry trade", "interest rate arbitrage",
    "swing trading", "day trading", "scalping",
    "buy and hold", "dollar cost averaging",
    "short selling", "shorting", "going short", "covering",
    "margin", "margin call", "leverage ratio",

    # Financial Instruments (Specific)
    "convertible bond", "callable bond", "putable bond",
    "zero-coupon bond", "floating rate note",
    "asset-backed security", "ABS", "mortgage-backed security", "MBS",
    "collateralized debt obligation", "CDO",
    "credit default swap", "CDS",
    "interest rate cap", "floor", "collar",
    "warrant", "rights offering",

    # Market Structure
    "primary market", "secondary market",
    "over-the-counter", "OTC market",
    "clearinghouse", "central counterparty", "CCP",
    "settlement", "T+2", "delivery versus payment",
    "circuit breaker", "trading halt",
    "price discovery", "market microstructure",

    # Quantitative Finance
    "quantitative analysis", "quant", "quantitative model",
    "algorithmic strategy", "trading algorithm",
    "backtesting", "backtest", "walk-forward analysis",
    "optimization", "parameter tuning",
    "statistical arbitrage", "stat arb",
    "factor model", "multi-factor model",
    "momentum factor", "value factor", "quality factor",

    # Corporate Finance
    "capital structure", "optimal capital structure",
    "cost of capital", "cost of equity", "cost of debt",
    "weighted average cost of capital", "WACC",
    "dividend policy", "share buyback", "stock repurchase",
    "debt financing", "equity financing",
    "project finance", "infrastructure financing",
    "financial distress", "bankruptcy", "Chapter 11",

    # Performance Metrics
    "return on equity", "ROE", "return on assets", "ROA",
    "return on invested capital", "ROIC",
    "net profit margin", "operating margin", "gross margin",
    "asset turnover", "inventory turnover",
    "debt-to-equity ratio", "leverage ratio",
    "interest coverage ratio", "debt service coverage",

    # Market Indicators
    "VIX", "volatility index", "fear index",
    "advance-decline line", "breadth indicator",
    "put-call ratio",
    "moving average", "exponential moving average", "EMA",
    "relative strength index", "RSI",
    "MACD", "moving average convergence divergence",
    "Bollinger bands", "support level", "resistance level",
]

# =============================================================================
# COMBINED FINANCIAL VOCABULARY
# =============================================================================

FINANCIAL: List[str] = FINANCIAL_BROAD + FINANCIAL_NARROW + FINANCIAL_SPECIFIC

# Statistics
print(f"Financial Vocabulary Statistics:")
print(f"  Tier 1 (Broad):    {len(FINANCIAL_BROAD)} terms")
print(f"  Tier 2 (Narrow):   {len(FINANCIAL_NARROW)} terms")
print(f"  Tier 3 (Specific): {len(FINANCIAL_SPECIFIC)} terms")
print(f"  Total:             {len(FINANCIAL)} terms")
print(f"\nDistribution:")
print(f"  Broad:    {len(FINANCIAL_BROAD)/len(FINANCIAL)*100:.1f}%")
print(f"  Narrow:   {len(FINANCIAL_NARROW)/len(FINANCIAL)*100:.1f}%")
print(f"  Specific: {len(FINANCIAL_SPECIFIC)/len(FINANCIAL)*100:.1f}%")
