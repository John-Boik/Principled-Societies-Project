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
    'person_member_spending_to_nonmember_SB_TS',
    'person_member_spending_to_nonmember_NP_pct',
    'person_member_spending_to_nonmember_NP_TS',
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
    'flexible_person_member_spending_to_nonmember_SB_TS',
    'flexible_person_member_spending_to_nonmember_NP_pct',
    'flexible_person_member_spending_to_nonmember_NP_TS',
    'flexible_person_nonmember_spending_to_member_SB_pct',
    'flexible_person_nonmember_spending_to_member_NP_pct',
    'flexible_person_nonmember_spending_to_member_PB_pct',
    'flexible_person_nonmember_spending_to_nonmember_SB_pct',
    'flexible_person_nonmember_spending_to_nonmember_NP_pct'
    ]


  data = {}
  for w in widgets:
    if w[0:9] == 'flexible_':
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
  
  
  
  #send info  
  #req = ajax.ajax()
  #req.bind('complete', complete_submit)
  #req.open('POST', '/runModel', True)
  #req.send(data)
  


 

##########################################################################################
# Bindings, etc on load
##########################################################################################


document['submit'].bind('click', submit_model)






