import torch

def calculate_tv_loss(generated_sequence):
    """Model 2: Generic Total Variation Loss (Smooths everything)"""
    step_differences = torch.abs(generated_sequence[:, 1:] - generated_sequence[:, :-1])
    return torch.mean(step_differences)

def calculate_monotonicity_loss(generated_sequence):
    """Model 3: Just the Monotonicity Prior"""
    step_differences = generated_sequence[:, 1:] - generated_sequence[:, :-1]
    monotonic_violations = torch.relu(step_differences)
    return torch.mean(monotonic_violations ** 2)

def calculate_all_physics_loss(generated_sequence):
    """Model 4: All 3 Lightweight Physical Priors"""
    loss = 0.0
    step_differences = generated_sequence[:, 1:] - generated_sequence[:, :-1]
    
    # 1. Monotonicity
    monotonic_violations = torch.relu(step_differences)
    loss += torch.mean(monotonic_violations ** 2) 
    
    # 2. Cliffs (Max Degradation Rate)
    drops = -step_differences
    cliff_violations = torch.relu(drops - 0.1) 
    loss += torch.mean(cliff_violations ** 2)
    
    # 3. Bounds
    upper_bounds = torch.relu(generated_sequence - 1.0)
    lower_bounds = torch.relu(-1.0 - generated_sequence)
    loss += torch.mean(upper_bounds ** 2) + torch.mean(lower_bounds ** 2)
    
    return loss

def count_monotonicity_violations(generated_sequence):
    step_differences = generated_sequence[:, 1:] - generated_sequence[:, :-1]
    violations = (step_differences > 1e-4).sum().item()
    return violations