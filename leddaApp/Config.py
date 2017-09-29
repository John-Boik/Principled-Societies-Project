
"""
This config file holds parameters for creating the HDF5 synthetic population & income file. Some
of these parameters are also used in calculating initial flows in setup_model.py.

If a variable is a percentage, use the term '_pct' or '_rate' in its name. These are converted
to fractions in setup_model.py.

If any parameters are changed, a new HDF5 file should be created.  To do this, run the script 
make_HDF5_pop_income.py in the additional_scripts folder.
"""

population = 50000
labor_participation_rate = 65.0
NP_reg_donation_pct_dollar_income = 2.0
WF_pct_NP_initial = 7.0
employed_income_Census_threshold = 10050
nonprofit_income_Census_threshold = 250000
 
gov_subsidy_SB_pct_total_income = 6.0
gov_contract_SB_pct_total_income = 1.6
gov_grant_NP_pct_total_income = 1.3
gov_contract_NP_pct_total_income = 1.7

gov_tax_standard_deduction = 2500
gov_tax_rate = 19.2

org_leakage_pct_revenue = 40.0

structural_unemployment_rate = 1.0
income_target_end_minimum = 21600

 
