from mmengine.registry import Registry

LOGGER = Registry('logger', locations=['finagentlight'])
LLM = Registry('llm', locations=['finagentlight'])
METRICS = Registry('metrics', locations=['finagentlight'])
DATASET = Registry('dataset', locations=['finagentlight'])
ENVIRONMENT = Registry('environment', locations=['finagentlight'])
SCALER = Registry('scaler', locations=['finagentlight'])
AGENT = Registry('agent', locations=['finagentlight'])
