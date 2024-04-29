import torch
from .utils import plot_tensor_histogram

def calculate_weight_diff(a, b):
    return a - b

def calculate_model_diffs(model_a, model_b):
    model_a_dict = model_a.state_dict()
    model_b_dict = model_b.state_dict()
    model_diffs = {}
    for key in model_a_dict.keys():
        if key in model_b_dict:
            model_diffs[key] = calculate_weight_diff(model_a_dict[key], model_b_dict[key])
            print(f"Diff calculated for {key}")
    return model_diffs

def calculate_sigmoid_ratios(base_model, target_model, epsilon=1e-6):
    sigmoid_ratios = {}
    target_diff = calculate_model_diffs(target_model, base_model)
    for key in target_diff.keys():
        diff_tensor = abs(target_diff[key])
        diff_min = diff_tensor.min().item()
        diff_max = diff_tensor.max().item()
        print(f"Key: {key}")
        print(f"  Diff Min: {diff_min}")
        print(f"  Diff Max: {diff_max}")

        # Uncomment the following lines to display a histogram of the tensor distribution
        # plot_tensor_histogram(diff_tensor, f"Histogram of differences for {key}", "Difference", "Frequency")
        
        if abs(diff_max - diff_min) < epsilon:
            print(f"  All values are the same. Setting sigmoid_diff to 0.")
            sigmoid_diff = torch.zeros_like(diff_tensor)
        else:
            normalized_diff = (diff_tensor - diff_min) / (diff_max - diff_min)
            sigmoid_diff = torch.sigmoid(normalized_diff * 12 - 6)
        sigmoid_ratios[key] = sigmoid_diff
        print(f"  Sigmoid Diff Min: {sigmoid_diff.min().item()}")
        print(f"  Sigmoid Diff Max: {sigmoid_diff.max().item()}")
    return sigmoid_ratios

def apply_model_diffs(target_model, model_diffs, sigmoid_ratios):
    target_state_dict = target_model.state_dict()
    for key in model_diffs.keys():
        print(key)
        print(model_diffs[key])
        ratio = sigmoid_ratios[key]
        print(ratio)
        scaled_diff = model_diffs[key] * (1 - ratio)
        target_state_dict[key] += scaled_diff
        print(f"Diff applied for {key}")
    target_model.load_state_dict(target_state_dict)