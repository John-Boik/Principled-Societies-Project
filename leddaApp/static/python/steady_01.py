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
  
  
  
  #send info  
  #req = ajax.ajax()
  #req.bind('complete', complete_submit)
  #req.open('POST', '/runModel', True)
  #req.send(data)
  


 

##########################################################################################
# Bindings, etc on load
##########################################################################################


document['submit'].bind('click', submit_model)


document['earmark_NP_donation'].value = "{:.4f}".format(0.0640304374712 *100)
document['earmark_NP_donation_TS'].value = "{:.4f}".format(0.513357363129 *100)
document['earmark_PB_subsidy'].value = "{:.4f}".format(0.0582408160466 *100)
document['earmark_PB_subsidy_TS'].value = "{:.4f}".format(0.480125590258 *100)
document['earmark_SB_subsidy'].value = "{:.4f}".format(0.0674546791704 *100)
document['earmark_SB_subsidy_TS'].value = "{:.4f}".format(0.564426856028 *100)
document['earmark_nurture'].value = "{:.4f}".format(0.362423961499 *100)
document['earmark_nurture_TS'].value = "{:.4f}".format(0.365616096143 *100)
document['person_member_spending_to_member_NP_TS'].value = "{:.4f}".format(0.420169330948 *100)
document['person_member_spending_to_member_NP_pct'].value = "{:.4f}".format(0.297577781158 *100)
document['person_member_spending_to_member_PB_TS'].value = "{:.4f}".format(0.413949588892 *100)
document['person_member_spending_to_member_PB_pct'].value = "{:.4f}".format(0.321834057452 *100)
document['person_member_spending_to_member_SB_TS'].value = "{:.4f}".format(0.463610294288 *100)
document['person_member_spending_to_member_SB_pct'].value = "{:.4f}".format(0.221268757038 *100)
document['person_member_spending_to_nonmember_NP_pct'].value = "{:.4f}".format(0.00569626363779 *100)
document['person_member_spending_to_nonmember_SB_pct'].value = "{:.4f}".format(0.153623140714 *100)
document['person_nonmember_spending_to_member_NP_pct'].value = "{:.4f}".format(0.0175785057973 *100)
document['person_nonmember_spending_to_member_PB_pct'].value = "{:.4f}".format(0.448028409727 *100)
document['person_nonmember_spending_to_member_SB_pct'].value = "{:.4f}".format(0.0309914441048 *100)
document['person_nonmember_spending_to_nonmember_NP_pct'].value = "{:.4f}".format(0.230608462683 *100)
document['person_nonmember_spending_to_nonmember_SB_pct'].value = "{:.4f}".format(0.272793177688 *100)



