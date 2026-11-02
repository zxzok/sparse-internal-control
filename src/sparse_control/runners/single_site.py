"""L15.H0 single-site reference: fixed coefficients, P/F directions."""
def run_single_site(root,config,profile,seeds,stage,output):
 from .fixed_intervention import evaluate_fixed_intervention
 from .preservation import run_preservation
 if stage=='preservation':return run_preservation(root,config,profile,seeds,output,task='refusal',single_site=True)
 return evaluate_fixed_intervention(root,config,profile,seeds,stage,output)
