from browser import document, alert, html


##########################################################################################
# Submit Model
##########################################################################################

def submit_model(evt):
  """
  run model
  """
  
  global results_counter
  elem = document['scenario']
  scenario = [opt.value for opt in elem.options if opt.selected==True][0]

  # from successful fun, fit=60
  data_01 = {
    'workforce_partition_SB': [33],
    'workforce_partition_NP': [34],
    'workforce_partition_PB': [33],
    'TSI': [35],
    'family_income_target_final': [110000],
    'population': [10000],
    'unemployment_rate': [5],
        
    'earmark_NP_donation': [5.114815312442174],
    'earmark_NP_donation_TS': [58.939234386483776],
    'earmark_PB_subsidy': [8.899206723284683],
    'earmark_PB_subsidy_TS': [53.163211852380456],
    'earmark_SB_subsidy': [8.51445110839744],
    'earmark_SB_subsidy_TS': [42.10429226478416],
    'earmark_nurture': [36.339790526860824],
    'earmark_nurture_TS': [36.439476630262504],
    'person_member_spending_to_member_NP_TS': [64.42464795292429],
    'person_member_spending_to_member_NP_pct': [23.505064474876036],
    'person_member_spending_to_member_PB_TS': [30.669838242642715],
    'person_member_spending_to_member_PB_pct': [27.55634331961967],
    'person_member_spending_to_member_SB_TS': [38.103915020305585],
    'person_member_spending_to_member_SB_pct': [32.604851195547866],
    'person_member_spending_to_nonmember_NP_pct': [0.5214587471568934],
    'person_member_spending_to_nonmember_SB_pct': [15.812282274675196],
    'person_nonmember_spending_to_member_NP_pct': [9.31615100786286],
    'person_nonmember_spending_to_member_PB_pct': [37.16891914950032],
    'person_nonmember_spending_to_member_SB_pct': [20.619519215802725],
    'person_nonmember_spending_to_nonmember_NP_pct': [23.946729130924936],
    'person_nonmember_spending_to_nonmember_SB_pct': [8.948681495909167],
    
    'CBFS_spending_ratio': [40]
  }


  data_02 = {
    'workforce_partition_SB': [33],
    'workforce_partition_NP': [34],
    'workforce_partition_PB': [33],
    'TSI': [35],
    'family_income_target_final': [110000],
    'population': [10000],
    'unemployment_rate': [5],
        
    'earmark_NP_donation': [8.644500319160668],
    'earmark_NP_donation_TS': [55.88660802799319],
    'earmark_PB_subsidy': [11.826420774518366],
    'earmark_PB_subsidy_TS': [48.02541415773954],
    'earmark_SB_subsidy': [12.696246181318719],
    'earmark_SB_subsidy_TS': [38.17651051378081],
    'earmark_nurture': [36.740916221721456],
    'earmark_nurture_TS': [36.04162220721133],
    'person_member_spending_to_member_NP_TS': [63.21892370652468],
    'person_member_spending_to_member_NP_pct': [21.87129068105028],
    'person_member_spending_to_member_PB_TS': [28.125704507011594],
    'person_member_spending_to_member_PB_pct': [28.685788956242465],
    'person_member_spending_to_member_SB_TS': [37.72741674077401],
    'person_member_spending_to_member_SB_pct': [33.326018279227966],
    'person_member_spending_to_nonmember_NP_pct': [0.6742319206138517],
    'person_member_spending_to_nonmember_SB_pct': [15.442610131504908],
    'person_nonmember_spending_to_member_NP_pct': [9.31615100786286],
    'person_nonmember_spending_to_member_PB_pct': [37.16891914950032],
    'person_nonmember_spending_to_member_SB_pct': [20.619519215802725],
    'person_nonmember_spending_to_nonmember_NP_pct': [23.946729130924936],
    'person_nonmember_spending_to_nonmember_SB_pct': [8.948681495909167],
    
    'CBFS_spending_ratio': [60]
  }


  data_03 = {
    'workforce_partition_SB': [12],
    'workforce_partition_NP': [70],
    'workforce_partition_PB': [18],
    'TSI': [35],
    'family_income_target_final': [110000],
    'population': [10000],
    'unemployment_rate': [5],

    'earmark_NP_donation': [31.444121377803235],
    'earmark_NP_donation_TS': [36.861424591974824],
    'earmark_PB_subsidy': [6.563161196207168],
    'earmark_PB_subsidy_TS': [45.794137921181914],
    'earmark_SB_subsidy': [2.878362271328804],
    'earmark_SB_subsidy_TS': [46.08915867561319],
    'earmark_nurture': [37.03198710972938],
    'earmark_nurture_TS': [35.758370560381316],
    'person_member_spending_to_member_NP_TS': [58.5754588219535],
    'person_member_spending_to_member_NP_pct': [41.64559556445119],
    'person_member_spending_to_member_PB_TS': [30.49837652360747],
    'person_member_spending_to_member_PB_pct': [20.00977132048462],
    'person_member_spending_to_member_SB_TS': [37.992360096672066],
    'person_member_spending_to_member_SB_pct': [22.651236070566835],
    'person_member_spending_to_nonmember_NP_pct': [0.26797928981835045],
    'person_member_spending_to_nonmember_SB_pct': [15.425513618400476],
    'person_nonmember_spending_to_member_NP_pct': [9.31615100786286],
    'person_nonmember_spending_to_member_PB_pct': [37.16891914950032],
    'person_nonmember_spending_to_member_SB_pct': [20.619519215802725],
    'person_nonmember_spending_to_nonmember_NP_pct': [23.946729130924936],
    'person_nonmember_spending_to_nonmember_SB_pct': [8.948681495909167],


    
    'CBFS_spending_ratio': [70]
  }


  data_04 = {
    'workforce_partition_SB': [12],
    'workforce_partition_NP': [70],
    'workforce_partition_PB': [18],
    'TSI': [35],
    'family_income_target_final': [110000],
    'population': [100000],
    'unemployment_rate': [5],

    'earmark_NP_donation': [31.443822806024002],
    'earmark_NP_donation_TS': [36.861381377187826],
    'earmark_PB_subsidy': [6.564268184025609],
    'earmark_PB_subsidy_TS': [45.79425475782429],
    'earmark_SB_subsidy': [2.8785491321809564],
    'earmark_SB_subsidy_TS': [46.08910397191154],
    'earmark_nurture': [37.03037421999677],
    'earmark_nurture_TS': [35.75840787354483],
    'person_member_spending_to_member_NP_TS': [58.575197309863356],
    'person_member_spending_to_member_NP_pct': [41.64537166504013],
    'person_member_spending_to_member_PB_TS': [30.498782117143026],
    'person_member_spending_to_member_PB_pct': [20.00984497040647],
    'person_member_spending_to_member_SB_TS': [37.99227586201974],
    'person_member_spending_to_member_SB_pct': [22.65120553757495],
    'person_member_spending_to_nonmember_NP_pct': [0.2680566449872311],
    'person_member_spending_to_nonmember_SB_pct': [15.42551646292275],
    'person_nonmember_spending_to_member_NP_pct': [9.31615100786286],
    'person_nonmember_spending_to_member_PB_pct': [37.16891914950032],
    'person_nonmember_spending_to_member_SB_pct': [20.619519215802725],
    'person_nonmember_spending_to_nonmember_NP_pct': [23.946729130924936],
    'person_nonmember_spending_to_nonmember_SB_pct': [8.948681495909167],

    'CBFS_spending_ratio': [70]
  }


  if scenario == '1':
    data = data_01 
  elif scenario == '2':
    data = data_02   
  elif scenario == '3':
    data = data_03  
  elif scenario == '4':
    data = data_04  
  else:
    raise Exception
    
  extra = {
    'doBFGS': [''],
    'doGenetic': [''],
    'doRandomStart': [''],
    
    'earmarks_TS_lb': ['35'],  # equal to TSI
    'earmarks_TS_ub': ['99'],
    
    'spending_TS_lb': ['20'],
    'spending_TS_ub': ['80'],
    
    'spending_pct_lb': ['.1'],
    'spending_pct_ub': ['70'],
    
    'flexible_earmark_NP_donation': [''],
    'flexible_earmark_NP_donation_TS': [''],
    'flexible_earmark_PB_subsidy': [''],
    'flexible_earmark_PB_subsidy_TS': [''],
    'flexible_earmark_SB_subsidy': [''],
    'flexible_earmark_SB_subsidy_TS': [''],
    'flexible_earmark_nurture': [''],
    'flexible_earmark_nurture_TS': [''],
    
    'flexible_person_member_spending_to_member_NP_TS': [''],
    'flexible_person_member_spending_to_member_NP_pct': [''],
    'flexible_person_member_spending_to_member_PB_TS': [''],
    'flexible_person_member_spending_to_member_PB_pct': [''],
    'flexible_person_member_spending_to_member_SB_TS': [''],
    'flexible_person_member_spending_to_member_SB_pct': [''],
    'flexible_person_member_spending_to_nonmember_NP_pct': [''],
    'flexible_person_member_spending_to_nonmember_SB_pct': [''],
    
    'flexible_person_nonmember_spending_to_member_NP_pct': [''],
    'flexible_person_nonmember_spending_to_member_PB_pct': [''],
    'flexible_person_nonmember_spending_to_member_SB_pct': [''],
    'flexible_person_nonmember_spending_to_nonmember_NP_pct': [''],
    'flexible_person_nonmember_spending_to_nonmember_SB_pct': ['']
    }
  
  data.update(extra)

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





