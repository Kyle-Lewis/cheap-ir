
def get_mixed_pruning_schedule(
    intermediate_hidden_dim,
    target_ffn_sparsity,
    num_warmup_steps,
    num_training_steps,
    num_prune_cycles,
):
    """ Create a pruning schedule that will result in the target sparsity within
        the provided number of cycles, and which will remove one head per cycle.

    The schedule is a list of dicts which indicate what action to take when the
    training run reaches certain step thresholds.

    example: [
        {"step": 100, "action": "accumulate"},
        {"step": 110, "action": "prune", "neurons_to_prune": 50},
        {"step": 200, "action": "accumulate"},
        {"step": 210, "action": "prune", "neurons_to_prune": 45},
        ...
    ]
    """
    
    schedule = []
    
    steps_per_cycle = (num_training_steps - num_warmup_steps) / num_prune_cycles
    accumulation_steps = max(min(100, int(steps_per_cycle / 10)), 1)

    target_dim = int((1.0 - target_ffn_sparsity) * intermediate_hidden_dim)
    div = (intermediate_hidden_dim - target_dim) // num_prune_cycles
    mod = (intermediate_hidden_dim - target_dim) % num_prune_cycles
    mod_sum = 0.0

    total_pruned = 0

    for prune_cycle in range(num_prune_cycles - 1):

        prune_step = int(num_warmup_steps + prune_cycle * steps_per_cycle)
        acc_step = prune_step - accumulation_steps

        neurons_to_prune = div
        mod_sum += mod / num_prune_cycles
        if mod_sum >= 1.0:
            neurons_to_prune += 1
            mod_sum -= 1.0
        total_pruned += neurons_to_prune
        
        schedule.append({
            "step": acc_step,
            "action": "accumulate",
        })

        schedule.append({
            "step": prune_step,
            "action": "prune",
            "neurons_to_prune": neurons_to_prune
        })

    ### Final step
    final_prune_cycle = num_prune_cycles - 1
    prune_step = int(num_warmup_steps + final_prune_cycle * steps_per_cycle)
    acc_step = prune_step - accumulation_steps

    remaining = intermediate_hidden_dim - total_pruned
    neurons_to_prune = remaining - target_dim
    
    schedule.append({
        "step": acc_step,
        "action": "accumulate",
    })

    schedule.append({
        "step": prune_step,
        "action": "prune",
        "neurons_to_prune": neurons_to_prune
    })

    return schedule