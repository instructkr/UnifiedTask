# AutoTaskTransfer (Under Construction)

AutoTaskTransfer is a Python package for automatic task transfer in neural networks using diff-based techniques.
This approach is conceived by [Jeonghwan Park (maywell)](https://github.com/StableFluffy), the main committer of InstructKR.

## Learn More
* You can check the example of ipynb here: https://github.com/StableFluffy/EasyLLMFeaturePorter

## Installation
`pip install AutoTaskTransfer`


```python
from AutoTaskTransfer.diff_transfer import calculate_model_diffs, calculate_sigmoid_ratios, apply_model_diffs
from AutoTaskTransfer.models.llama import load_llama_model

informative_model = load_llama_model("gradientai/Llama-3-8B-Instruct-262k")
base_model = load_llama_model("kuotient/Meta-Llama-3-8B-Instruct")
target_model = load_llama_model("beomi/Llama-3-Open-Ko-8B-Instruct-preview")

model_diffs = calculate_model_diffs(informative_model, base_model)
sigmoid_ratios = calculate_sigmoid_ratios(base_model, target_model)
apply_model_diffs(target_model, model_diffs, sigmoid_ratios)
...
```

## License
This project is licensed under the MIT License.
