from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_MARKET_INTELLIGENCE_DESCRIPTION = """Provide a comprehensive summary of market intelligence and extract the most valuable key insights to support decision-making.
** Asset Information **
* You are currently focusing on summarizing and extracting the key insights of the market intelligence of a $$asset_type$$ known as $$asset_name$$, which is denoted by the symbol $$asset_symbol$$.
* This $$asset_type$$ is publicly traded and is listed on the $$asset_exchange$$. Its primary operations are within the $$asset_sector$$ sector, specifically within the $$asset_industry$$ industry.
* To provide you with a better understanding, here is a brief description of $$asset_name$$: $$asset_description$$.

** Target **
As an analyst, your target is to summarize the market intelligence for $$asset_symbol$$ using the following comprehensive information:

** Market Intelligence **
The following market intelligence (e.g., news, financial reports) contains latest (i.e., today) information related to $$asset_symbol$$, including the corresponding dates, headlines, and contents, with each item distinguished by a unique ID. Furthermore, if the day is not closed for trading, the section also provides the open, high, low, close, and adjusted close prices.
$$market_intelligence_content$$

** Potential Impact of Market Intelligence **
* Irrelevant Market Intelligence:
    - Ignore market intelligence unrelated to asset prices, such as advertisements on news platforms.
* Effect Duration:
    - Short-term: Impacts asset prices for a few days.
    - Medium-term: Influences asset prices for a few weeks.
    - Long-term: Affects asset prices for several months.
    - Unclear duration: Treat as long-term.
* Market Sentiment:
    - Positive: Focus on favorable effects (e.g., boosting confidence, increasing demand) but consider risks (e.g., overreaction, temporary effects).
    - Negative: Focus on unfavorable effects (e.g., panic, reputation damage) but acknowledge benefits (e.g., market correction, investment insights).
    - Neutral: No clear positive or negative impact; uncertain sentiment should be treated as neutral.
* Collaborators and Competitors:
    - Intelligence related to collaborators or competitors can influence asset prices.
* Recency of Information:
    - Prioritize recent market intelligence over older information, as past events have less effect on the present.
"""

_MARKET_INTELLIGENCE_PARAMETER_ANALYSIS_DESCRIPTION = """This field is used to extract key insights from the above information. You should analyze step-by-step and follow the rules as follows and do not miss any of them:
* You MUST disregard UNRELATED market intelligence.
* For each piece of market intelligence, you should analyze it and extract key insights according to the following steps:
    - Extract the key insights that can represent this market intelligence. It should NOT contain IDs, $$asset_name$$ or $$asset_symbol$$.
    - Analyze the market effects duration and provide the duration of the effects on asset prices. You are only allowed to select the only one of the three types: SHORT-TERM, MEDIUM-TERM and LONG-TERM.
    - Analyze the market sentiment and provide the type of market sentiment. A clear preference over POSITIVE or NEGATIVE is much better than being NEUTRAL. You are only allowed to select the only one of the three types: POSITIVE, NEGATIVE and NEUTRAL.
* The analysis you provide for each piece of market intelligence should be concise and clear, with no more than 40 tokens per piece.
* Your analysis MUST be in the following format:
    - ID: 000001 - Analysis that you provided for market intelligence 000001.
    - ID: 000002 - Analysis that you provided for market intelligence 000002.
    - ...

** Example **
(the analysis starts here)
- ID: 000001 - Analysis that you provided for market intelligence 000001.
- ID: 000002 - Analysis that you provided for market intelligence 000002.
...
(the analysis ends here)
"""

_MARKET_INTELLIGENCE_PARAMETER_SUMMARY_DESCRIPTION = """This field is used to summarize the above analysis and extract key investment insights. You should summarize step-by-step and follow the rules as follows and do not miss any of them:
* You MUST disregard UNRELATED market intelligence.
* Because this field is primarily used for decision-making in trading tasks, you should focus primarily on asset related key investment insights.
* Please combine and summarize market intelligence on similar sentiment tendencies and duration of effects on asset prices.
* You should provide an overall analysis of all the market intelligence, explicitly provide a market sentiment (POSITIVE, NEGATIVE or NEUTRAL) and provide a reasoning for the analysis.
* Summary that you provided for market intelligence should contain IDs (e.g., ID: 000001, 000002).
* The summary you provide should be concise and clear, with no more than 300 tokens.
"""

MarketIntelligenceTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='market_intelligence',
        description=_MARKET_INTELLIGENCE_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'market_intelligence_analysis': {
                    'type': 'string',
                    'description': _MARKET_INTELLIGENCE_PARAMETER_ANALYSIS_DESCRIPTION,
                },
                'market_intelligence_summary': {
                    'type': 'string',
                    'description': _MARKET_INTELLIGENCE_PARAMETER_SUMMARY_DESCRIPTION,
                },
            },
            'required': ['market_intelligence_analysis', 'market_intelligence_summary'],
        },
    ),
)

_DECISION_MAKING_DESCRIPTION = """Make SELL, HOLD, and BUY decisions based on comprehensive news information.
** Asset Information **
* You are currently targeting the trading of a company known as $$asset_name$$, which is denoted by the symbol $$asset_symbol$$. This corporation is publicly traded and is listed on the $$asset_exchange$$. Its primary operations are within the $$asset_sector$$ sector, specifically within the $$asset_industry$$ industry.
* To provide you with a better understanding, here is a brief description of $$asset_name$$: $$asset_description$$.

** Target **
In this role, your objective is to make correct trading decisions during the trading process of the asset represented by the $$asset_symbol$$, and considering step by step about the decision reasoning. To do so effectively, you will rely on a comprehensive set of information and data as follows.

** Trader Preference **
As a risk-seeking trader, you actively seek higher risk and may pursue market volatility and high fluctuations. You are willing to take on more significant risks in pursuit of potentially higher returns and may employ aggressive trading strategies.

** Market Intelligence Summary**
The following is a summary of the market intelligence analysis you have conducted. It contains the key insights extracted from the market intelligence, focusing on the sentiment and duration of effects on asset prices. Each item is distinguished by a unique ID.
$$market_intelligence_summary$$

Based on the above information, you should step-by-step analyze the summary of the market intelligence. And provide the reasoning for what you should to BUY, SELL or HOLD on the asset.
"""

_DECISION_MAKING_PARAMETER_ANALYSIS_DESCRIPTION = """You should analyze step-by-step how the above information may affect the results of your decisions. You need to follow the rules as follows and do not miss any of them:
* When analyzing the summary of market intelligence, you should determine whether the market intelligence are positive, negative or neutral.
    - If the overall is neurtal, your decision should pay less attention to the summary of market intelligence.
    - If the overall is positive or negative. you should give a decision result based on this.
* When analyzing the analysis of price movements, you should determine whether the future trend is bullish or bearish and reflect on the lessons you've learned.
    - If the future trend is bullish, you should consider a BUY instead of a HOLD to increase your profits.
    - If the future trend is bearish, you should consider a SELL instead of a HOLD to prevent further losses.
    - You should provide your decision result based on the analysis of price movements.
* When analyzing the analysis of the past trading decisions, you should reflect on the lessons you've learned.
    - If you have missed a BUY opportunity, you should BUY as soon as possible to increase your profits.
    - If you have missed a SELL, you should SELL immediately to prevent further losses.
    - You should provide your decision result based on the reflection of the past trading decisions.
* When analyzing the professional investment guidances, you should determine whether the guidances show the trend is bullish or bearish. And provide your decision results.
* When analyzing the decisions and explanations of some trading strategies, you should consider the results and explanations of their decisions together. And provide your decision results.
* When providing the final decision, you should pay less attention to the market intelligence whose sentiment is neutral or unrelated.
* When providing the final decision, you should pay more attention to the market intelligence which will cause an immediate impact on the price.
* When providing the final decision, if the overall market intelligence is mixed up, you should pay more attention to the professional investment guidances, and consider which guidance is worthy trusting based on historical price.
* Combining the results of all the above analysis and decisions, you should determine whether the current situation is suitable for BUY, SELL or HOLD. And provide your final decision results.
"""

_DECISION_MAKING_PARAMETER_REASONING_DESCRIPTION = """You should think step-by-step and provide the detailed reasoning to determine the decision result executed on the current observation for the trading task. Please strictly follow the following constraints and output formats:
* You should provide the reasoning for each point of the "analysis" and the final results you provide.
"""

_DECISION_MAKING_PARAMETER_DECISION_DESCRIPTION = """Based on the above information and your analysis. Please strictly follow the following constraints and output formats:
* You can only output one of BUY, HOLD and SELL.
* The above information may be in the opposite direction of decision-making (e.g., BUY or SELL), but you should consider step-by-step all of the above information together to give an exact BUY or SELL decision result.

** Example **
BUY
"""

DecisionMakingTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='decision_making',
        description=_DECISION_MAKING_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'decision_making_analysis': {
                    'type': 'string',
                    'description': _DECISION_MAKING_PARAMETER_ANALYSIS_DESCRIPTION,
                },
                'decision_making_reasoning': {
                    'type': 'string',
                    'description': _DECISION_MAKING_PARAMETER_REASONING_DESCRIPTION,
                },
                'decision_making_decision': {
                    'type': 'string',
                    'description': _DECISION_MAKING_PARAMETER_DECISION_DESCRIPTION,
                },
            },
            'required': [
                'decision_making_analysis',
                'decision_making_reasoning',
                'decision_making_decision',
            ],
        },
    ),
)

tools = [
    MarketIntelligenceTool,
    DecisionMakingTool,
]
