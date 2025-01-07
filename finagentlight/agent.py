import re
from copy import deepcopy
from typing import Any, Dict

import dotenv

dotenv.load_dotenv()

from finagentlight.function_calling import tools
from finagentlight.llm import LLM
from finagentlight.logger import logger
from finagentlight.registry import AGENT
from finagentlight.utils import json
from finagentlight.utils.file_utils import assemble_project_path
from finagentlight.utils.message import Message, TextContent


def convert_text(tool: Any, state: Dict[str, Any]) -> str:
    tool_string = json.dumps(tool)

    keys = state.keys()

    # find all keys in text
    exiting_keys = []
    for key in keys:
        if f'$${key}$$' in tool_string:
            exiting_keys.append(key)

    # get all values of exiting keys
    exiting_values = [state[key] for key in exiting_keys]

    # if there is None in exiting_values, return None
    if None in exiting_values:
        return ''
    else:
        for key in exiting_keys:
            if isinstance(state[key], list) or isinstance(state[key], dict):
                tool_string = tool_string.replace(
                    f'$${key}$$', json.dumps(state[key], indent=4)
                )
            else:
                tool_string = tool_string.replace(f'$${key}$$', str(state[key]))

    clean_string = re.sub(r'[\x00-\x1F\x7F]', '', tool_string)
    tool = json.loads(clean_string)
    return tool


@AGENT.register_module(force=True)
class Agent:
    def __init__(self, llm: LLM, system_prompt_path: str, user_prompt_path: str):
        self.llm = llm

        system_prompt_path = assemble_project_path(system_prompt_path)
        user_prompt_path = assemble_project_path(user_prompt_path)

        with open(system_prompt_path, 'r') as file:
            self.system_prompt = file.read()

        with open(user_prompt_path, 'r') as file:
            self.user_prompt = file.read()

        self.tools = tools

    def reset(self):
        self.last_action = 'No action taken.'
        logger.info('| Agent reset.')

    def step(self, state: Dict[str, Any]):
        params = self._get_params(state)

        response = self.llm.completion(**params)

        assistant_msg = response.choices[0].message

        tool_call = assistant_msg.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        tool_name = tool_call.function.name

        self.last_action = tool_name

        action: Dict[str, Any] = {
            'tool_name': tool_name,
        }
        action.update(arguments)

        return action

    def _convert_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        res_state: Dict[str, Any] = {}

        asset_name = state['asset_name']
        asset_symbol = state['asset_symbol']
        asset_exchange = state['asset_exchange']
        asset_sector = state['asset_sector']
        asset_industry = state['asset_industry']
        asset_description = state['asset_description']
        asset_type = state['asset_type']
        timestamp = state['timestamp']

        price = deepcopy(state['history_price'])
        news = deepcopy(state['history_news'])

        if len(news) > 20:
            news = news.sample(n=20)

        market_intelligence_content = f'Date: Today is {timestamp}.\n'

        if len(price) > 0:
            open = price['open'].values[0]
            high = price['high'].values[0]
            low = price['low'].values[0]
            close = price['close'].values[0]
            adj_close = price['adj_close'].values[0]
            market_intelligence_content += f'Prices: Open: ({open}), High: ({high}), Low: ({low}), Close: ({close}), Adj Close: ({adj_close})\n'
        else:
            market_intelligence_content += 'Prices: Today is closed for trading.\n'

        if len(news) == 0:
            market_intelligence_content = 'There is no latest market_intelligence.\n'
        else:
            latest_market_intelligence_list = []

            for row in news.iterrows():
                row = row[1]
                id = row['id']
                title = row['title']
                text = row['text']

                latest_market_intelligence_item = (
                    f'ID: {id}\n' + f'Headline: {title}\n' + f'Content: {text}\n'
                )

                latest_market_intelligence_list.append(latest_market_intelligence_item)

            if len(latest_market_intelligence_list) == 0:
                market_intelligence_content = (
                    'There is no latest market_intelligence.\n'
                )
            else:
                market_intelligence_content = '\n'.join(latest_market_intelligence_list)

        res_state.update(
            {
                'timestamp': timestamp,
                'asset_name': asset_name,
                'asset_type': asset_type,
                'asset_symbol': asset_symbol,
                'asset_exchange': asset_exchange,
                'asset_sector': asset_sector,
                'asset_industry': asset_industry,
                'asset_description': asset_description,
                'market_intelligence_content': market_intelligence_content,
            }
        )

        return res_state

    def _convert_tools(self, state: Dict[str, Any]):
        tools = []
        for tool in self.tools:
            tool = convert_text(tool, state)
            tools.append(tool)
        return tools

    def _get_params(self, state: Dict[str, Any]):
        messages: list[Message] = [
            Message(
                role='system',
                content=[
                    TextContent(
                        text=self.system_prompt,
                    )
                ],
            )
        ]
        example_message = self.user_prompt
        if example_message:
            messages.append(
                Message(
                    role='user',
                    content=[TextContent(text=example_message)],
                )
            )

        last_action_message = Message(
            role='user',
            content=[
                TextContent(text=f'The last action you chose was {self.last_action}')
            ],
        )
        messages.append(last_action_message)

        state = self._convert_state(state)
        tools = self._convert_tools(state)

        params = {
            'messages': self.llm.format_messages_for_llm(messages),
            'tools': tools,
        }
        return params
