from browser import document, alert, html








##########################################################################################
# Submit Model
##########################################################################################

def submit_model(evt):
  """
  run model
  """
  
  global results_counter
  
  widgets = [
    'population',
    'family_income_target_final',
    'TSI',
    'unemployment_rate',
    'workforce_partition_SB',
    'workforce_partition_NP',
    'workforce_partition_PB',
    
    'earmark_SB_subsidy',
    'earmark_PB_subsidy',
    'earmark_NP_donation',
    'earmark_nurture',
    'earmark_SB_subsidy_TS',
    'earmark_PB_subsidy_TS',
    'earmark_NP_donation_TS',
    'earmark_nurture_TS',
    'person_member_spending_to_member_SB_pct',
    'person_member_spending_to_member_SB_TS',
    'person_member_spending_to_member_NP_pct',
    'person_member_spending_to_member_NP_TS',
    'person_member_spending_to_member_PB_pct',
    'person_member_spending_to_member_PB_TS',
    'person_member_spending_to_nonmember_SB_pct',
    'person_member_spending_to_nonmember_NP_pct',
    'person_nonmember_spending_to_member_SB_pct',
    'person_nonmember_spending_to_member_NP_pct',
    'person_nonmember_spending_to_member_PB_pct',
    'person_nonmember_spending_to_nonmember_SB_pct',
    'person_nonmember_spending_to_nonmember_NP_pct',
    
    'flexible_earmark_SB_subsidy',
    'flexible_earmark_PB_subsidy',
    'flexible_earmark_NP_donation',
    'flexible_earmark_nurture',
    'flexible_earmark_SB_subsidy_TS',
    'flexible_earmark_PB_subsidy_TS',
    'flexible_earmark_NP_donation_TS',
    'flexible_earmark_nurture_TS',
    'flexible_person_member_spending_to_member_SB_pct',
    'flexible_person_member_spending_to_member_SB_TS',
    'flexible_person_member_spending_to_member_NP_pct',
    'flexible_person_member_spending_to_member_NP_TS',
    'flexible_person_member_spending_to_member_PB_pct',
    'flexible_person_member_spending_to_member_PB_TS',
    'flexible_person_member_spending_to_nonmember_SB_pct',
    'flexible_person_member_spending_to_nonmember_NP_pct',
    'flexible_person_nonmember_spending_to_member_SB_pct',
    'flexible_person_nonmember_spending_to_member_NP_pct',
    'flexible_person_nonmember_spending_to_member_PB_pct',
    'flexible_person_nonmember_spending_to_nonmember_SB_pct',
    'flexible_person_nonmember_spending_to_nonmember_NP_pct',
    'earmarks_TS_lb',
    'earmarks_TS_ub',
    'spending_TS_lb',
    'spending_TS_ub',
    'doBFGS',
    'doGenetic',
    'doRandomStart'
    ]


  data = {}
  
  # set upper and lower bounds if widgits are empty
  for w in ['earmarks_TS_lb', 'spending_TS_lb']:
    if document[w].value.strip() == '':
      document[w].value = '0'
  for w in ['earmarks_TS_ub', 'spending_TS_ub']:
    if document[w].value.strip() == '':
      document[w].value = '100'   
  
  # set check boxes to 0/1, and otherwise get values for widgits
  for w in widgets:
    if (w[0:9] in ['flexible_']) or (w in ['doGenetic', 'doBFGS','doRandomStart']):
      data[w] = document[w].checked
    else:
      data[w] = document[w].value
  
  # create form to post data
  document['hidden_form'].html = ""
  keys = data.keys()
  method = "post"
  form = html.FORM(method='post', action='/runModel', target="_blank")
  for k in keys:
    ele = html.INPUT(type="hidden", name=k, value=data[k])
    form <= ele
  
  document['hidden_form'] <= form

  form.submit();  


 

##########################################################################################
# Bindings, etc on load
##########################################################################################


document['submit'].bind('click', submit_model)

# from successful fun, fit=60
tmp = [
  ['workforce_partition_SB', 33],
  ['workforce_partition_NP', 34],
  ['workforce_partition_PB', 33],
  ['TSI', 35],
  ['family_income_target_final', 110000],
  ['population', 10000],
  
  ['earmark_NP_donation', 6.402547170210858],
  ['earmark_NP_donation_TS', 51.344221799135305],
  ['earmark_PB_subsidy', 5.825170718608573],
  ['earmark_PB_subsidy_TS', 48.011991172761164],
  ['earmark_SB_subsidy', 6.745779498862711],
  ['earmark_SB_subsidy_TS', 56.44172097708899],
  ['earmark_nurture', 36.24497570556464],
  ['earmark_nurture_TS', 36.56068565383187],
  ['person_member_spending_to_member_NP_TS', 42.01583698771104],
  ['person_member_spending_to_member_NP_pct', 29.75717246854228],
  ['person_member_spending_to_member_PB_TS', 41.394765355348376],
  ['person_member_spending_to_member_PB_pct', 32.18280145355416],
  ['person_member_spending_to_member_SB_TS', 46.36061188684639],
  ['person_member_spending_to_member_SB_pct', 22.126619909351383],
  ['person_member_spending_to_nonmember_NP_pct', 0.5700925153256662],
  ['person_member_spending_to_nonmember_SB_pct', 15.363313653226507],
  ['person_nonmember_spending_to_member_NP_pct', 1.757619973128018],
  ['person_nonmember_spending_to_member_PB_pct', 44.81176707379108],
  ['person_nonmember_spending_to_member_SB_pct', 3.098604908159066],
  ['person_nonmember_spending_to_nonmember_NP_pct', 23.057099211271552],
  ['person_nonmember_spending_to_nonmember_SB_pct', 27.274908833650287]

  ]

for name, val in tmp:
  if name in   ['workforce_partition_SB', 'workforce_partition_NP', 'workforce_partition_PB', 
    'TSI', 'family_income_target_final', 'population']:
    document[name].value = "{:d}".format(val)
  else:
    document[name].value = "{:.4f}".format(val)




