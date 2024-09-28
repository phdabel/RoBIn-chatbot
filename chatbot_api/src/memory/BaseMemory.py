from typing import Any, Dict
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

class ModifiedConversationBufferMemory(ConversationBufferMemory):

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        if isinstance(input_str, Dict):
            input_str = input_str.get('query_text')
        self.chat_memory.add_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )

    async def asave_context(
            self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        if isinstance(input_str, Dict):
            input_str = input_str.get('query_text')
        await self.chat_memory.aadd_messages(
            [HumanMessage(content=input_str), AIMessage(content=output_str)]
        )