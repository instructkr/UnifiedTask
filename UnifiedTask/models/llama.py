from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llama_model(model_name):
    """
    Load a Llama model using the specified model name.

    Args:
        model_name (str): The name or path of the Llama model to load.

    Returns:
        model (AutoModelForCausalLM): The loaded Llama model.
        tokenizer (AutoTokenizer): The tokenizer associated with the loaded model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer