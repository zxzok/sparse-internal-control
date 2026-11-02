"""Explicit position rules; padding side does not silently choose a rule."""
def resolve_position(attention_mask,rule="last_valid"):
    import torch
    mask=torch.as_tensor(attention_mask)
    if mask.ndim!=2 or torch.any(mask.sum(1)==0):raise ValueError("Nonempty two-dimensional mask required")
    if rule=="sum_mask_minus_one":return mask.sum(1).long()-1
    if rule=="last_valid":return (torch.arange(mask.shape[1],device=mask.device)[None,:]*mask).max(1).values.long()
    if rule=="last_tensor":return torch.full((mask.shape[0],),mask.shape[1]-1,device=mask.device,dtype=torch.long)
    raise ValueError(f"Unknown position rule {rule}")
def gather_at_positions(values,positions):
    from .transformerlens import _gather_at_positions
    return _gather_at_positions(values,positions)
def scatter_add_at_positions(values,positions,delta):
    from .transformerlens import _scatter_add_at_positions
    return _scatter_add_at_positions(values,positions,delta)
