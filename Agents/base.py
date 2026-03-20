# agents/base.py
import os
from config.settings import transit_config

def _demo_mode() -> bool:
    return not bool(transit_config.openai_api_key.strip().startswith("sk-"))

def call_llm(system_prompt: str, user_prompt: str, demo_response: str = "") -> str:
    if _demo_mode():
        return f"[DEMO] {demo_response}"
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatOpenAI(model=transit_config.model_name,
                         temperature=transit_config.temperature,
                         api_key=transit_config.openai_api_key)
        r = llm.invoke([SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)])
        return r.content
    except Exception as e:
        return f"[LLM ERROR: {e}]\n{demo_response}"
