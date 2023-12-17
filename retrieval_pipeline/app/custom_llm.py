from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from llama_index.indices.prompt_helper import PromptHelper
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# define prompt helper
# set maximum input size
max_input_size = 1024
# set number of output tokens
num_output = 256 
prompt_helper = PromptHelper(max_input_size, num_output)

class CustomLLM(LLM):
    model_name = "gpt-2-117M"
    model_path = "./models/gpt-2-117M/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        pass   

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        pass 
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    @torch.inference_mode()
    def generate_response(self, prompt: str, max_new_tokens=num_output, temperature=0.7, top_k=1, top_p=1.0):
        encoded_prompt = self.tokenizer.encode(prompt, return_tensors='pt')
        max_length = len(encoded_prompt[0]) + max_new_tokens
        with torch.no_grad():
            output = self.model.generate(encoded_prompt, 
                                         max_length=max_length,
                                         temperature=temperature, 
                                         top_k=top_k, 
                                         top_p=top_p, 
                                         do_sample=True)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = self.generate_response(prompt)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"