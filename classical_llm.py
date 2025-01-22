from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ClassicalLLM:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", max_new_tokens=500):
        """
        Initializes the Llama 3 model and tokenizer using Hugging Face's Transformers library.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for performance
            device_map="auto"           # Automatically map layers to devices
        )
        self.max_new_tokens = max_new_tokens

        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            # Assign `eos_token` as the `pad_token`
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, context, query, do_sample=True):
        """
        Generates a response using Llama 3 given a context and query.
        
        :param context: Retrieved context or document text.
        :param query: User's question or query.
        :param do_sample: Whether to use sampling-based generation.
        :return: Generated answer as a string.
        """
        # Construct the input prompt
        prompt = f"Context: {context}\n\nQuery: {query}\nAnswer:"

        # Tokenize the prompt with padding and truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Set generation parameters based on `do_sample`
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": 0.7,  # Adjust for response variability
                "top_p": 0.9,        # Nucleus sampling for diversity
            })
        else:
            generation_kwargs["do_sample"] = False

        # Generate the response using the model
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_kwargs
        )

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract and return only the answer portion
        if "Answer:" in generated_text:
            return generated_text.split("Answer:")[-1].strip()
        return generated_text
