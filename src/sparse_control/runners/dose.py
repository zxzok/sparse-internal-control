"""Dose evaluation of explicitly saved controls."""
from .fixed_intervention import evaluate_fixed_intervention
def run_dose_scan(root,config,profile,seeds,output,interventions=None):
    return evaluate_fixed_intervention(root,config,profile,seeds,"dose",output,interventions)
