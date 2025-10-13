import os
import requests
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
import json


class NetAppLLM(LLM):
    """Custom LLM wrapper for NetApp LLM Proxy API."""
    
    model: str = "gpt-5"
    user: str = "pg47711"
    api_url: str = "https://llm-proxy-api.ai.eng.netapp.com/v1/completions"
    authorization_token: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get token from environment if not provided
        if not self.authorization_token:
            self.authorization_token = os.getenv("NETAPP_LLM_TOKEN")
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "netapp_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the NetApp LLM API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.authorization_token}"
        }
        
        data = {
            "model": self.model,
            "user": self.user,
            "prompt": prompt
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the completion text from the response
            # Handle various response formats
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                return choice.get("text", choice.get("message", {}).get("content", ""))
            elif "text" in result:
                return result["text"]
            elif "content" in result:
                return result["content"]
            else:
                # Debug: print the actual response structure
                print(f"Unexpected response format: {result}")
                return str(result)
        except Exception as e:
            raise


class ChatNetAppLLM(BaseChatModel):
    """Chat model wrapper for NetApp LLM Proxy API."""
    
    model: str = "gpt-5"
    user: str = "pg47711"
    api_url: str = "https://llm-proxy-api.ai.eng.netapp.com/v1/chat/completions"
    authorization_token: Optional[str] = None
    bound_tools: List[dict] = []
    debug: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get token from environment if not provided
        if not self.authorization_token:
            self.authorization_token = os.getenv("NETAPP_LLM_TOKEN")
    
    def bind_tools(self, tools: List[BaseTool], **kwargs):
        """Bind tools to the model."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.__class__(
            model=self.model,
            user=self.user,
            api_url=self.api_url,
            authorization_token=self.authorization_token,
            bound_tools=formatted_tools,
            debug=self.debug,
            **kwargs
        )
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "chat_netapp_llm"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from messages."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.authorization_token}"
        }
        
        # Convert messages to OpenAI chat format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Include tool calls if present
                ai_msg = {"role": "assistant", "content": msg.content or ""}
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # Convert back to OpenAI format
                    ai_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"])
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                openai_messages.append(ai_msg)
            elif isinstance(msg, ToolMessage):
                # Add tool results
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
        
        data = {
            "model": self.model,
            "user": self.user,
            "messages": openai_messages
        }
        
        # Add tools if bound
        if self.bound_tools:
            data["tools"] = self.bound_tools
            data["tool_choice"] = "auto"
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the completion text and tool calls from the response
            content = ""
            tool_calls = []
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message_data = choice.get("message", {})
                content = message_data.get("content", choice.get("text", ""))
                
                # Check for tool calls
                if "tool_calls" in message_data:
                    tool_calls = message_data["tool_calls"]
                    if self.debug:
                        print(f"[DEBUG] Found {len(tool_calls)} tool calls!")
            elif "text" in result:
                content = result["text"]
            elif "content" in result:
                content = result["content"]
            else:
                content = str(result)
            
            # Create AIMessage with tool calls if present
            # LangGraph expects tool_calls in a specific format
            if tool_calls:
                # Format tool calls for LangGraph
                formatted_tool_calls = []
                for tc in tool_calls:
                    formatted_tool_calls.append({
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"]),
                        "id": tc["id"]
                    })
                
                message = AIMessage(
                    content=content or "",
                    tool_calls=formatted_tool_calls
                )
            else:
                message = AIMessage(content=content or "")
            
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise
    
