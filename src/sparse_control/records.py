"""Typed measurement identities and shape validation."""
from dataclasses import dataclass
from math import isfinite, prod

def finite(values, name):
    if not values or any(v is None or not isfinite(v) for v in values): raise ValueError(f"{name}: expected nonempty finite values")
def unique_ids(records, key="record_id"):
    keys=[r[key] for r in records]
    if len(set(keys))!=len(keys): raise ValueError(f"Duplicate {key}")
    return True
@dataclass(frozen=True)
class InputRecord:
    input_id: str
    prompt_ids: list
    input_ids: list
    attention_mask: list
    readout_positions: list
    def validate(self):
        n=len(self.prompt_ids)
        if not self.input_id or len(set(self.prompt_ids))!=n: raise ValueError("Input identity/prompt order")
        if not (len(self.input_ids)==len(self.attention_mask)==len(self.readout_positions)==n): raise ValueError("Batch dimension mismatch")
        for ids,mask,p in zip(self.input_ids,self.attention_mask,self.readout_positions):
            if len(ids)!=len(mask) or p<0 or p>=len(ids) or mask[p]!=1: raise ValueError("Invalid readout position")
        return self
@dataclass(frozen=True)
class InterventionRecord:
    intervention_id: str
    candidates: list
    coefficients: list
    direction_mode: str
    nominal_dose: float=1.0
    def validate(self):
        if len(self.candidates)!=len(self.coefficients): raise ValueError("Support/coefficient length mismatch")
        if len(set(self.candidates))!=len(self.candidates): raise ValueError("Duplicate candidates")
        if any(not isfinite(x) for x in self.coefficients) or not isfinite(self.nominal_dose): raise ValueError("Nonfinite intervention")
        if self.direction_mode not in ("P","F","class_means"): raise ValueError("Unknown direction condition")
        return self
@dataclass(frozen=True)
class ReadoutRecord:
    record_id: str
    intervention_id: str
    prompt_ids: list
    readout_shape: list
    source_behavior: list
    target_behavior: list
    forward_behavior: list
    def validate(self):
        n=prod(self.readout_shape)
        if n<=0 or self.readout_shape[0]!=len(self.prompt_ids): raise ValueError("Readout shape/prompt order mismatch")
        for k in ("source_behavior","target_behavior","forward_behavior"):
            a=getattr(self,k);finite(a,k)
            if len(a)!=n: raise ValueError(f"{k}: shape mismatch")
        return self
@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    experiment_id: str
    model_asset_id: str
    execution_profile_id: str
    resolved_config: dict
    completed: bool
