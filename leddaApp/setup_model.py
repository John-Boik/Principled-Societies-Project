
import numpy as np
import tables as tb
from scipy import stats
import os, pdb

from leddaApp import app
from leddaApp import Config



#########################################################################################
# Classes Used
#########################################################################################


class Object(object):
  """
  A generic class that holds data, parameters, and variables
  """
  
  def __init__(self, *dic, **kwargs):
    self.flexibleList = []
    self.Variance = .001
    # if *dict is passed
    for d in dic:
      for k in d.keys():
        value = d[k]
        self.__dict__[k] = value
    # if key-value pairs are passed  
    for k in kwargs.keys():
      value = kwargs[k]
      self.__dict__[k] = value

  def __setitem__(self, key, value):
    self.__dict__[key] = value

  def __getitem__(self, key):
    return self.__dict__[key]

  def set_flexible(self):
    # if doOptimization, then select flexible variables
    keys = list(self.__dict__.keys())
    keys.sort()
    flexibleList = []
    for k in keys:
      if k[0:9] == 'flexible_':
        flexibleList.append(k[9:])
    self.__dict__['flexibleList'] = flexibleList
    



#########################################################################################
# Setup Function
#########################################################################################

def setup(data):
  """
  This is the main function to run the interactive steady state model. It creates the HDF5 
  file that contains synthetic population and income data, or reads from a previously created
  HDF5 file. It also sets up model parameters and calls a fitness function. It returns fitness data.
  """
  
  Print = False
  
  if Print:
    print("\nin setup")

    

  """
  TSI =  0.35
  earmark_NP_donation =  0.08
  earmark_NP_donation_TS =  0.35
  earmark_nurture =  0.37
  earmark_nurture_TS =  0.35
  earmark_PB_subsidy =  0.1
  earmark_PB_subsidy_TS =  0.35
  earmark_subsidy_SB =  0.06
  earmark_subsidy_SB_TS =  0.35
  family_income_target_final =  110000.0
  person_member_pct_spending_to_member_NP =  0.2
  person_member_pct_spending_to_member_PB =  0.2
  person_member_pct_spending_to_member_SB =  0.2
  person_member_pct_spending_to_nonmember_NP =  0.2
  person_member_pct_spending_to_nonmember_SB =  0.2
  person_member_spending_to_member_NP_TS =  0.35
  person_member_spending_to_member_SB_TS =  0.35
  person_member_spending_to_nonmember_NP_TS =  0.35
  person_member_spending_to_nonmember_SB_TS =  0.35
  person_nonmember_pct_spending_to_member_NP =  0.2
  person_nonmember_pct_spending_to_member_PB =  0.2
  person_nonmember_pct_spending_to_member_SB =  0.2
  person_nonmember_pct_spending_to_nonmember_NP =  0.2
  person_nonmember_pct_spending_to_nonmember_SB =  0.2
  unemployment_rate =  0.05
  workforce_partition_NP =  0.34
  workforce_partition_PB =  0.33
  workforce_partition_SB =  0.33


  """  
  
  # initialize the Data object X that holds general info and serves as container for other objects
  X = Object(data) 
  X.Config = Object() 
  
  if X.doOptimization:
    # create list of flexible variables for minimization
    X.set_flexible()
  
  # add parameters in Config.py to object X.Config
  keys = dir(Config)
  
  if Print:
    print("\nConfig:")  
  for key in keys:
    if key[0] == "_":
      continue
    value = Config.__dict__[key]
    if ('_pct' in key) or ('_rate' in key):
      value = value/100.
    if Print:
      print(key, "= ", value)
    X.Config.__setitem__(key, value)
  
  """
  NP_reg_donation_pct_dollar_income =  2.0
  WF_pct_NP_initial =  7.0
  employed_income_Census_threshold =  10050
  gov_contract_NP_pct_total_income =  1.7
  gov_contract_SB_pct_total_income =  1.6
  gov_grant_NP_pct_total_income =  1.3
  gov_subsidy_SB_pct_total_income =  6.0
  gov_tax_rate =  19.2
  gov_tax_standard_deduction =  2500
  income_target_end_minimum =  21600
  labor_participation_rate =  65.0
  org_leakage_pct_revenue =  40.0
  population =  100000
  structural_unemployment_rate =  1.0
  """
  
  # note that Config.population (to make synthetic population data) may be different 
  # from X.population (a model parameter)
  X.populationRatio = X.population / X.Config.population
  
  # create some needed meta variables
  X.number_employed = np.round(X.population * \
    X.Config.labor_participation_rate * (1-X.unemployment_rate), 0).astype('i')
  
  X.number_unemployed = np.round(X.population * \
    X.Config.labor_participation_rate * X.unemployment_rate, 0).astype('i')
  
  X.number_NILF = np.round(X.population * \
    (1-X.Config.labor_participation_rate), 0).astype('i')
  
  # adjust if rounding error
  if X.population - (X.number_employed + 
    X.number_unemployed + X.number_NILF) == 1:
      X.number_unemployed += 1
  if X.population - (X.number_employed + 
    X.number_unemployed + X.number_NILF) == -1:
      X.number_unemployed -= 1     

  assert np.allclose(X.population, X.number_employed + \
    X.number_unemployed + X.number_NILF)

  # open synthetic population file
  dataFolder = app.config['DATA_FOLDER']
  fn = os.path.join(dataFolder, 'steady_state_population.hdf5')
  HDF5 = tb.open_file(fn, mode='r')
  
  tableID = None
  for tableName in HDF5.root._v_children:
    if np.allclose(
      HDF5.root._v_children[tableName].attrs['unemployment_rate'], 
      X.unemployment_rate*100):
      tableID = tableName.split('_')[-1]
      break
  assert tableID is not None
  
  # attach persons table (TP) and families table (TF) to object X
  X.TP = HDF5.root._v_children['Persons_'  + tableID]
  X.TF = HDF5.root._v_children['Families_' + tableID]
  
  # setup initial funds for government
  X = setGovRates(X)

  # initialize some Ledda and county arrays
  incomeP = X.TP.cols.R_wages_NP_dollars[:] + X.TP.cols.R_wages_SB_dollars[:] + \
    X.TP.cols.R_gov_support_dollars[:]
  #X.R_dollars = incomeP.sum() * X.populationRatio
  
  # collect county information
  countyInfo(X)
  
  # make dict to hold counts and initial/final target incomes for 9 types of persons
  stocksDic = make_stocks(X)
  
  # make dict to hold summary of initial/final counts
  countsDic = make_counts(X, stocksDic)
  
  # return histogram data
  histoDic = {'hist': X.TF.attrs.hist, 'bin_edges': X.TF.attrs.bin_edges}

  return X, stocksDic, countsDic, histoDic


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def setGovRates(X):
  """
  
  Initialize government spending amounts and the tax rate. Taxes are based on adjusted gross income 
  (AGI).  Every person donates dollars to NP at a fixed rate, and AGI equals total income - max(standard 
  deduction, amount donated to nonprofits).  Any AGI < 0 is set to 0.
  
  Every year, forprofits and nonprofits receive the same level of government spending due to grants, 
  contracts, and subsidies.  This is conservative. Government payments do not increase, even if
  the local economy grows. 

  """
  
  Print = False
  
  if Print:
    print("""

    # =====================================================================
    # Initialize government spending amounts and tax rate
    # ===================================================================== 

    """)
  
  Gov = Object()
  
  # grants to nonprofits
  W = X.TP.get_where_list("(work_status==4)")
  P_grant_NP_dollars = X.TP.cols.R_dollars[:].sum() * X.Config.gov_grant_NP_pct_total_income
  fraction = P_grant_NP_dollars / X.TP.cols.R_wages_NP_dollars[:].sum()
  P_grant_NP_dollars = P_grant_NP_dollars * X.populationRatio
  
  if Print:
    print("\nGov spending, NP grants, fraction NP wages = {0:,.4g}".format(fraction))
    print("Gov spending, NP grants = ${0:,.9g}".format(
      np.round(P_grant_NP_dollars, 0).item()))
  
  Gov.grant_NP_dollars_annual = P_grant_NP_dollars

  # contracts with NP
  W = X.TP.get_where_list("(work_status==4)")
  P_contract_NP_dollars = X.TP.cols.R_dollars[:].sum() * \
    X.Config.gov_contract_NP_pct_total_income
  fraction = P_contract_NP_dollars / X.TP.cols.R_wages_NP_dollars[:].sum()
  P_contract_NP_dollars = P_contract_NP_dollars * X.populationRatio
  
  if Print:
    print("\nGov spending, NP contracts, fraction NP wages = {0:,.4g}".format(fraction))
    print("Gov spending, NP contracts = ${0:,.9g}".format(
      np.round(P_contract_NP_dollars,0).item()))
  
  Gov.contract_NP_dollars_annual = P_contract_NP_dollars

  # contracts with forprofits
  W = X.TP.get_where_list("(work_status==5)")
  P_contract_forprofit_dollars = X.TP.cols.R_dollars[:].sum() * \
    X.Config.gov_contract_SB_pct_total_income
  fraction = P_contract_forprofit_dollars / X.TP.cols.R_wages_SB_dollars[:].sum()
  P_contract_forprofit_dollars = P_contract_forprofit_dollars * X.populationRatio
  
  if Print:
    print(
      "\nGov spending, forprofit contracts, fraction forprofit wages = {0:,.4g}".format(fraction))
    print(
      "Gov spending, forprofit contracts = ${0:,.9g}".format(
        np.round(P_contract_forprofit_dollars,0).item()))
  
  Gov.contract_forprofit_dollars_annual = P_contract_forprofit_dollars

  # subsidies to forprofits
  W = X.TP.get_where_list("(work_status==5)")
  P_subsidy_forprofit_dollars = X.TP.cols.R_dollars[:].sum() * \
    X.Config.gov_subsidy_SB_pct_total_income
  fraction = P_subsidy_forprofit_dollars / X.TP.cols.R_wages_SB_dollars[:].sum()
  P_subsidy_forprofit_dollars = P_subsidy_forprofit_dollars * X.populationRatio
  
  if Print:
    print(
      "\nGov spending, forprofit subsidy, fraction forprofit wages = {0:,.4g}".format(fraction))
    print(
      "Gov spending, forprofit subsidy = ${0:,.9g}".format(np.round(P_subsidy_forprofit_dollars,0).item()))
  
  Gov.subsidy_forprofit_dollars_annual = P_subsidy_forprofit_dollars   
    
  # NIWF & unemployed
  W = X.TP.get_where_list("(work_status==0)|(work_status==2)")
  P_support_dollars = X.TP.cols.R_dollars[:][W].sum() 
  fraction = P_support_dollars / X.TP.cols.R_dollars[:].sum()
  P_support_dollars = P_support_dollars * X.populationRatio
  
  if Print:
    print(
      "\nGov spending, NIWF & unemployed support, fraction total income = {0:,.4g}".format(fraction))
    print(
      "Gov spending, NIWF & unemployed support = ${0:,.9g}".format(np.round(P_support_dollars,0).item()))
    
  # mean support for NIWF & unemployed
  Gov.support_dollars_mean = P_support_dollars / (float(W.size) * X.populationRatio) 
  
  # assume every person donates to NPs at fixed rate, based on dollar income
  donation_dollars = X.TP.cols.R_dollars[:] * X.Config.NP_reg_donation_pct_dollar_income
  AGI = np.maximum(0, X.TP.cols.R_dollars[:] - np.maximum(
    donation_dollars, X.Config.gov_tax_standard_deduction))
    
  spending_total = P_grant_NP_dollars + P_contract_NP_dollars + P_contract_forprofit_dollars + \
    P_subsidy_forprofit_dollars + P_support_dollars  
  
  Gov.tax_rate = spending_total / (AGI.sum() * X.populationRatio)
  
  if Print:
    print("\nGovernment tax rate = {0:,.4g}\n".format(Gov.tax_rate))

  X.Gov = Gov
  
  return X


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def countyInfo(X):
  """
  Summaries of initial county income data.
  """

  if Print:
    print("""
    
    # =====================================================================
    # Initial County Data
    # ===================================================================== 
    
    """)
  
  incomeP = X.TP.cols.R_wages_NP_dollars[:] + X.TP.cols.R_wages_SB_dollars[:] + \
    X.TP.cols.R_gov_support_dollars[:]
  
  if Print:
    print("\nfamily income, median = ${0:,.9g}".format(
      np.round(np.median(X.TF.cols.R_dollars[:]),0).item()))
  
    print("family income, mean = ${0:,.9g}".format(
      np.round(X.TF.cols.R_dollars[:].mean(),0).item()))
  
  We1 = X.TP.get_where_list("(work_status >=4)")
  ave_working_income = incomeP[We1].mean()
  
  if Print:
    print("\nmean working income = ${0:,.9g}".format(np.round(ave_working_income,0).item()))
  
  ave_income = incomeP.mean()
  
  if Print:
    print(("\nmean person income = ${0:,.9g}, percentile of person income at " + 
      "mean = {1:,.9g}").format(
      np.round(ave_income,0).item(), stats.percentileofscore(incomeP, ave_income)))
  
  W0 = np.where(incomeP < ave_income)[0]
  W1 = np.where(incomeP >= ave_income)[0]
  
  if Print:
    print("total person income <  mean person income: ${0:,.9g}, size= {1:,d}".format(
      np.round(incomeP[W0].sum() * X.populationRatio,0).item(),  int(W0.size * X.populationRatio)))
    
    print("total person income >= mean person income: ${0:,.9g}, size= {1:,d}".format(
      np.round(incomeP[W1].sum() * X.populationRatio,0).item(), int(W1.size * X.populationRatio)))
    
    print("total county income = ${0:,.9g}\n".format(
      np.round(incomeP.sum() * X.populationRatio,0).item()))
    
    for i in np.linspace(0,100,21):
      print("  percentile = {0:>5.4g}, person income = ${1:>12,.9g}".format(
        i, np.round(stats.scoreatpercentile(incomeP, i),0).item()))
    print("\n")
  
  famRecTot = X.TF.cols.R_dollars[:]
  totalNIWF = float(X.TP.get_where_list("((work_status==0) | (work_status==1))").size) 
  totalUnemp = float(X.TP.get_where_list("((work_status==2) | (work_status==3))").size)
  
  if 1==2:
    # for testing, otherwise takes too long
    for i in np.linspace(0,100,21):
      famIncCut = np.round(stats.scoreatpercentile(famRecTot, i),0).item()
      Wc = np.where(famRecTot <= famIncCut)[0]
      countNIWF = 0
      countUnemp = 0
      for wc in Wc:
        fid = X.TF.cols.fid[wc]
        wfid = X.TP.get_where_list("fid=={0:d}".format(fid))
        assert len(wfid) == 2
        for pid in wfid:
          if X.TP.cols.work_status[pid] in [0,1]:
            countNIWF += 1
          if X.TP.cols.work_status[pid] in [2,3]:
            countUnemp += 1          
      fNIWF = countNIWF / totalNIWF
      fUnemp = countUnemp / totalUnemp       
          
      print(
        ("  percentile = {0:>5.4g}, family income = ${1:>12,.9g},  fract of NIWF = {2:.4f},  " + \
          "fract of Unemp = {3:.4f}").format(i, famIncCut, fNIWF, fUnemp))
    print("\n")

  # calculate thresholds for family income target. Could increase family income target 
  # by 3% to ensure that all families choose Wage Option 1 (and not token bonus, Wage Option 2) 
  threshold_family = X.family_income_target_final  
  threshold_person = threshold_family/2.
  
  persons_below_threshold = checkFamilyIncome(X, 0, threshold_family, Membership=0)
  
  percentile_threshold_person = stats.percentileofscore(X.TP.cols.R_dollars[:], threshold_person)
  
  if Print:  
    print(("\nthreshold for membership, person = ${0:,.9g}, percentile of county "  + 
      "person income = {1:,.4g}").format(
      np.round(threshold_person, 0).item(), percentile_threshold_person))
  
  X.percentile_threshold_family = stats.percentileofscore(X.TF.cols.R_dollars[:], threshold_family)
  
  if Print:
    print(("threshold for membership, family = ${0:,.9g}, percentile of county " +
      "family income= {1:,.4g}").format(np.round(threshold_family, 0).item(), X.percentile_threshold_family))
  
  ws0 = X.TP.get_where_list("work_status==0")
  NIWF_below = np.intersect1d(persons_below_threshold, ws0, assume_unique=True)
  
  if Print:
    print("\nfraction of total NIWF below family threshold = {0:,.4g}".format(
      NIWF_below.size / float(ws0.size)))
    
    print("fraction below family threshold that are NIWF = {0:,.4g}".format(
      NIWF_below.size / float(persons_below_threshold.size)))
    
    print(("fraction below family threshold that are NIWF or unemployed (1% member " + 
      "unemployment) = {0:,.4g}").format((NIWF_below.size / float(persons_below_threshold.size)) + \
      (X.population * X.Config.labor_participation_rate * .01) / \
      float(persons_below_threshold.size)))

  ave_income_threshold = X.TP.cols.R_dollars[:][persons_below_threshold].mean()

  if Print:  
    print("\ntotal family income <= threshold_family = ${0:,.9g}".format(
      np.round(X.TP.cols.R_dollars[:][persons_below_threshold].sum() * X.populationRatio, 0).item()))
    
    print("mean of total family income <= threshold_family = ${0:,.9g}".format(
      np.round(ave_income_threshold,0).item()))
  
  #income_below_threshold = X.TP.cols.R_dollars[:][persons_below_threshold].sum()
  
  W4 = X.TP.get_where_list("(work_status >=4)")
  persons_below_threshold_working = np.intersect1d(W4,persons_below_threshold)
  ave_income_working_county_threshold = X.TP.cols.R_dollars[:][persons_below_threshold_working].mean()
  
  if Print:
    print("\naverage income, working, county, below threshold = ${0:,.9g}".format(
      np.round(ave_income_working_county_threshold,0).item()))
    
  Wnp = X.TP.get_where_list("(work_status ==4) | (work_status ==6)")  
  
  if Print:
    print(("average income nonprofit = ${0:,.9g}, fraction of county income = " + 
      "{1:,.4g}").format(
      np.round(X.TP.cols.R_dollars[:][Wnp].mean(), 0).item(), 
      X.TP.cols.R_dollars[:][Wnp].sum() / X.TP.cols.R_dollars[:].sum()))

  W3 = X.TP.get_where_list("(work_status <=3)")  
  
  if Print:
    print(("average income NIWF and unemployed = ${0:,.9g}, fraction of county " + 
      "income = {1:,.4g}\n").format(
      np.round(X.TP.cols.R_dollars[:][W3].mean(), 0).item(), 
      X.TP.cols.R_dollars[:][W3].sum() / X.TP.cols.R_dollars[:].sum()))       



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def make_stocks(X):
  """ 
  Create dict to hold initial and final incomes, counts for each of 9 person types 
  (9 work_status types)
    work_status = indicator of type of job/support:
      person_nonmember_NIWF         = 0
      person_member_NIWF            = 1
      person_nonmember_unemployed   = 2
      person_member_unemployed      = 3
      person_nonmember_NP           = 4
      person_nonmember_SB           = 5
      person_member_NP              = 6
      person_member_SB              = 7
      person_member_PB              = 8

  """
  
  Print = False
  
  
  stocksDic = {}
  
  FR = X.TF.cols.R_dollars[:]   # family received income
  PR = X.TP.cols.R_dollars[:]   # persons received income
  WS = X.TP.cols.work_status[:]
  
  if Print:
    print("\nunique WS: ", np.unique(WS))
  
  # the income target for families (those over the target will not become members)
  familyPercentileCut = np.round(
    stats.scoreatpercentile(FR, X.percentile_threshold_family),0).item()
      
  personPostCBFSIncomeTarget = familyPercentileCut / 2. 
  
  # dont allow target below living wage
  assert personPostCBFSIncomeTarget >= X.Config.income_target_end_minimum  
  
  stocksDic = {}
  stocksDic['familyPercentileCut'] = familyPercentileCut
  stocksDic['personPostCBFSIncomeTarget'] = personPostCBFSIncomeTarget
  
  stocksDic.update({'initial':{}, 'final':{}})
  
  # there are 9 types of persons, depending on income source
  # recall that number of families = popultion / 2
  LeddaF = np.where(FR <= familyPercentileCut)[0]
  NonLeddaF = np.where(FR > familyPercentileCut)[0]

  # convert familiy IDs to person IDs, considering work status
  LeddaPersons = np.hstack((X.TF.cols.person1[:][LeddaF], X.TF.cols.person2[:][LeddaF]))
  NonLeddaPersons = np.hstack((X.TF.cols.person1[:][NonLeddaF], X.TF.cols.person2[:][NonLeddaF]))
  
  if Print:
    print("\nLedda.size= ", LeddaPersons.size, " NonLedda.size= ", NonLeddaPersons.size)
  assert np.allclose( (LeddaPersons.size + NonLeddaPersons.size) * X.populationRatio, X.population)
  
  for ws in range(0,9):
    #print("\n--------- ws= ", ws)
    
    stocksDic['initial'][ws] = {}
    stocksDic['final'][ws] = {}
    
    wLedda = np.where(WS[LeddaPersons] == ws)[0]
    Ledda = LeddaPersons[wLedda]
    wNonLedda = np.where(WS[NonLeddaPersons] == ws)[0]
    NonLedda = NonLeddaPersons[wNonLedda]      
    
    if Ledda.size + NonLedda.size == 0:
      pctLedda = 0 
      pctNonLedda = 0
    else:
      pctLedda = Ledda.size / float(Ledda.size + NonLedda.size)
      pctNonLedda = NonLedda.size / float(Ledda.size + NonLedda.size)

    if Ledda.size == 0:
      meanLedda = 0 
      sumLedda = 0
    else:
      meanLedda = PR[Ledda].mean() 
      sumLedda =  PR[Ledda].sum() * X.populationRatio

    if NonLedda.size == 0:
      meanNonLedda = 0 
      sumNonLedda = 0
    else:
      meanNonLedda = PR[NonLedda].mean() 
      sumNonLedda =  PR[NonLedda].sum() * X.populationRatio 
      assert np.allclose(meanNonLedda, sumNonLedda / (NonLedda.size * X.populationRatio))

    # per-person, post CBFS T&D values 
    temp = {
      'cntLedda':     Ledda.size *    X.populationRatio, 
      'cntNonLedda':  NonLedda.size * X.populationRatio, 
      'pctLedda':pctLedda, 'pctNonLedda':pctNonLedda, 
      'meanLedda':meanLedda, 'meanNonLedda':meanNonLedda, 
      'sumLedda':sumLedda, 'sumNonLedda':sumNonLedda}
    
    stocksDic['initial'][ws].update(temp)
    assert np.allclose(
      stocksDic['initial'][ws]['sumNonLedda'], 
      stocksDic['initial'][ws]['meanNonLedda'] * stocksDic['initial'][ws]['cntNonLedda'])
    
    # placeholders for final counts/incomes
    temp = {'cntLedda':0, 'cntNonLedda':0, 'pctLedda':0, 'pctNonLedda':0, 'meanLedda':0, 
      'meanNonLedda':0, 'sumLedda':0, 'sumNonLedda':0}
    
    stocksDic['final'][ws].update(temp)

  
  # complete dict for final counts/incomes based on chosen targets
  # per-person, post CBFS T&D values
  stocksDic['final'][0].update({'cntNonLedda': stocksDic['initial'][0]['cntNonLedda']})
  stocksDic['final'][0].update({'pctNonLedda': 1.})
  stocksDic['final'][0].update({'meanNonLedda': stocksDic['initial'][0]['meanNonLedda']})
  stocksDic['final'][0].update({'sumNonLedda': stocksDic['initial'][0]['sumNonLedda']})
  
  assert np.allclose(stocksDic['final'][0]['sumNonLedda'], 
    stocksDic['final'][0]['meanNonLedda'] * stocksDic['final'][0]['cntNonLedda'])
  
  stocksDic['final'][1].update({'cntLedda': stocksDic['initial'][0]['cntLedda']})
  stocksDic['final'][1].update({'pctLedda': 1.})
  stocksDic['final'][1].update({'meanLedda': stocksDic['personPostCBFSIncomeTarget']})
  stocksDic['final'][1].update({'sumLedda': stocksDic['final'][1]['meanLedda'] \
    * stocksDic['final'][1]['cntLedda']})

  stocksDic['final'][2].update({'cntNonLedda': stocksDic['initial'][2]['cntNonLedda']})
  stocksDic['final'][2].update({'pctNonLedda': 1.})
  stocksDic['final'][2].update({'meanNonLedda': stocksDic['initial'][2]['meanNonLedda']})
  stocksDic['final'][2].update({'sumNonLedda': stocksDic['initial'][2]['sumNonLedda']})

  LeddaPop = stocksDic['initial'][0]['cntLedda'] + stocksDic['initial'][2]['cntLedda'] + \
    stocksDic['initial'][4]['cntLedda'] + stocksDic['initial'][5]['cntLedda']
  LeddaNIWF = stocksDic['final'][1]['cntLedda']
  LeddaWF = LeddaPop - LeddaNIWF
  
  stocksDic['final'][3].update({'cntLedda': int(round(
    LeddaWF * X.Config.structural_unemployment_rate))})
  stocksDic['final'][3].update({'pctLedda': 1.})
  stocksDic['final'][3].update({'meanLedda': stocksDic['personPostCBFSIncomeTarget']})
  stocksDic['final'][3].update({'sumLedda': stocksDic['final'][3]['meanLedda'] * \
    stocksDic['final'][3]['cntLedda']})
  
  stocksDic['final'][4].update({'cntNonLedda': stocksDic['initial'][4]['cntNonLedda']})
  stocksDic['final'][4].update({'pctNonLedda': 1.})
  stocksDic['final'][4].update({'meanNonLedda': stocksDic['initial'][4]['meanNonLedda']})
  stocksDic['final'][4].update({'sumNonLedda': stocksDic['initial'][4]['sumNonLedda']})
  
  stocksDic['final'][5].update({'cntNonLedda': stocksDic['initial'][5]['cntNonLedda']})
  stocksDic['final'][5].update({'pctNonLedda': 1.})
  stocksDic['final'][5].update({'meanNonLedda': stocksDic['initial'][5]['meanNonLedda']})
  stocksDic['final'][5].update({'sumNonLedda': stocksDic['initial'][5]['sumNonLedda']})    
  
  LeddaWF2 = stocksDic['initial'][4]['cntLedda'] + stocksDic['initial'][5]['cntLedda'] + \
    stocksDic['initial'][2]['cntLedda']
  assert np.allclose(LeddaWF2, LeddaWF) 
  
  stocksDic['LeddaWF'] = LeddaWF
  stocksDic['WF'] = LeddaWF + stocksDic['final'][2]['cntNonLedda'] + \
    stocksDic['final'][4]['cntNonLedda'] + stocksDic['final'][5]['cntNonLedda']
  
  stocksDic['LeddaPop'] = LeddaPop
  stocksDic['LeddaNIWF'] = LeddaNIWF
  pct_WF_employed = 1 - X.Config.structural_unemployment_rate
  
  stocksDic['final'][6].update({'cntLedda': int(round(
    LeddaWF * pct_WF_employed * X.workforce_partition_NP))})
  stocksDic['final'][6].update({'pctLedda': 1.})
  stocksDic['final'][6].update({'meanLedda': stocksDic['personPostCBFSIncomeTarget']})
  stocksDic['final'][6].update({'sumLedda': stocksDic['final'][6]['meanLedda'] * \
    stocksDic['final'][6]['cntLedda']})
  
  stocksDic['final'][7].update({'cntLedda': int(round(
    LeddaWF * pct_WF_employed * X.workforce_partition_SB))})
  stocksDic['final'][7].update({'pctLedda': 1.})
  stocksDic['final'][7].update({'meanLedda': stocksDic['personPostCBFSIncomeTarget']})
  stocksDic['final'][7].update({'sumLedda': stocksDic['final'][7]['meanLedda'] * \
    stocksDic['final'][7]['cntLedda']})

  cntLedda_1 = int(round(LeddaWF * pct_WF_employed * X.workforce_partition_PB))
  cntLedda_2 = LeddaWF * pct_WF_employed  - \
    (stocksDic['final'][6]['cntLedda'] + stocksDic['final'][7]['cntLedda'])
  
  assert np.allclose(cntLedda_1, cntLedda_2, atol=1)
  
  stocksDic['final'][8].update({'cntLedda': int(round(cntLedda_2))})
  stocksDic['final'][8].update({'pctLedda': 1.})
  stocksDic['final'][8].update({'meanLedda': stocksDic['personPostCBFSIncomeTarget']})
  stocksDic['final'][8].update({'sumLedda': stocksDic['final'][8]['meanLedda'] * \
    stocksDic['final'][8]['cntLedda']})
         
  
  initialPop = 0
  finalPop = 0
  for ws in range(9):
    initialPop += stocksDic['initial'][ws]['cntLedda'] + stocksDic['initial'][ws]['cntNonLedda']
    finalPop += stocksDic['final'][ws]['cntLedda'] + stocksDic['final'][ws]['cntNonLedda']
  
  if Print:
    print('X.population: ', X.population)
    print('initialPop: ', initialPop)
    print('finalPop: ', finalPop)
  
  assert np.allclose(round(initialPop), X.population, atol=1)
  assert np.allclose(round(finalPop), X.population, atol=1)
  
  if Print:  
    print("""
  
     ------------------------------------------------------------------------------------
Percentile = {:3.0f},   Family income cut for Ledda/NonLedda = {:,.0f} T&D  
  Final post-CBFS person income target = {:,.0f} T&D == {:,.0f} T&D per member family
All T&D values are post-CBFS, mean T&D values are per person 

Status                               Count                 Percent                 Mean                   Sum
                              Initial     Final      Initial     Final      Initial     Final      Initial     Final        
                            
0  NIWF-notLS      Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}

1  NIWF-LS         Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}

2  Unemp-notLS     Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}

3  Unemp-LS        Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}

4  Emp-notLFNJ-NP  Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}

5  Emp-notLFNJ-SB  Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}

6  Emp-LFNJ-NP     Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}

7  Emp-LFNJ-SB     Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}

8  Emp-LFNJ-PB     Ledda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      
                NonLedda:  {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}    {:9.4g} {:9.4g}      

""".format(X.percentile_threshold_family, stocksDic['familyPercentileCut'], stocksDic['personPostCBFSIncomeTarget'],
    stocksDic['personPostCBFSIncomeTarget'] * 2,
    
    stocksDic['initial'][0]['cntLedda'], stocksDic['final'][0]['cntLedda'], 
    stocksDic['initial'][0]['pctLedda'], stocksDic['final'][0]['pctLedda'], 
    stocksDic['initial'][0]['meanLedda'], stocksDic['final'][0]['meanLedda'],
    stocksDic['initial'][0]['sumLedda'], stocksDic['final'][0]['sumLedda'],
    stocksDic['initial'][0]['cntNonLedda'], stocksDic['final'][0]['cntNonLedda'], 
    stocksDic['initial'][0]['pctNonLedda'], stocksDic['final'][0]['pctNonLedda'], 
    stocksDic['initial'][0]['meanNonLedda'], stocksDic['final'][0]['meanNonLedda'],
    stocksDic['initial'][0]['sumNonLedda'], stocksDic['final'][0]['sumNonLedda'],  

    stocksDic['initial'][1]['cntLedda'], stocksDic['final'][1]['cntLedda'], 
    stocksDic['initial'][1]['pctLedda'], stocksDic['final'][1]['pctLedda'], 
    stocksDic['initial'][1]['meanLedda'], stocksDic['final'][1]['meanLedda'],
    stocksDic['initial'][1]['sumLedda'], stocksDic['final'][1]['sumLedda'],
    stocksDic['initial'][1]['cntNonLedda'], stocksDic['final'][1]['cntNonLedda'], 
    stocksDic['initial'][1]['pctNonLedda'], stocksDic['final'][1]['pctNonLedda'], 
    stocksDic['initial'][1]['meanNonLedda'], stocksDic['final'][1]['meanNonLedda'],
    stocksDic['initial'][1]['sumNonLedda'], stocksDic['final'][1]['sumNonLedda'],  
      
    stocksDic['initial'][2]['cntLedda'], stocksDic['final'][2]['cntLedda'], 
    stocksDic['initial'][2]['pctLedda'], stocksDic['final'][2]['pctLedda'], 
    stocksDic['initial'][2]['meanLedda'], stocksDic['final'][2]['meanLedda'],
    stocksDic['initial'][2]['sumLedda'], stocksDic['final'][2]['sumLedda'],
    stocksDic['initial'][2]['cntNonLedda'], stocksDic['final'][2]['cntNonLedda'], 
    stocksDic['initial'][2]['pctNonLedda'], stocksDic['final'][2]['pctNonLedda'], 
    stocksDic['initial'][2]['meanNonLedda'], stocksDic['final'][2]['meanNonLedda'],
    stocksDic['initial'][2]['sumNonLedda'], stocksDic['final'][2]['sumNonLedda'],  
      
    stocksDic['initial'][3]['cntLedda'], stocksDic['final'][3]['cntLedda'], 
    stocksDic['initial'][3]['pctLedda'], stocksDic['final'][3]['pctLedda'], 
    stocksDic['initial'][3]['meanLedda'], stocksDic['final'][3]['meanLedda'],
    stocksDic['initial'][3]['sumLedda'], stocksDic['final'][3]['sumLedda'],
    stocksDic['initial'][3]['cntNonLedda'], stocksDic['final'][3]['cntNonLedda'], 
    stocksDic['initial'][3]['pctNonLedda'], stocksDic['final'][3]['pctNonLedda'], 
    stocksDic['initial'][3]['meanNonLedda'], stocksDic['final'][3]['meanNonLedda'],
    stocksDic['initial'][3]['sumNonLedda'], stocksDic['final'][3]['sumNonLedda'],  
      
    stocksDic['initial'][4]['cntLedda'], stocksDic['final'][4]['cntLedda'], 
    stocksDic['initial'][4]['pctLedda'], stocksDic['final'][4]['pctLedda'], 
    stocksDic['initial'][4]['meanLedda'], stocksDic['final'][4]['meanLedda'],
    stocksDic['initial'][4]['sumLedda'], stocksDic['final'][4]['sumLedda'],
    stocksDic['initial'][4]['cntNonLedda'], stocksDic['final'][4]['cntNonLedda'], 
    stocksDic['initial'][4]['pctNonLedda'], stocksDic['final'][4]['pctNonLedda'], 
    stocksDic['initial'][4]['meanNonLedda'], stocksDic['final'][4]['meanNonLedda'],
    stocksDic['initial'][4]['sumNonLedda'], stocksDic['final'][4]['sumNonLedda'],  
      
    stocksDic['initial'][5]['cntLedda'], stocksDic['final'][5]['cntLedda'], 
    stocksDic['initial'][5]['pctLedda'], stocksDic['final'][5]['pctLedda'], 
    stocksDic['initial'][5]['meanLedda'], stocksDic['final'][5]['meanLedda'],
    stocksDic['initial'][5]['sumLedda'], stocksDic['final'][5]['sumLedda'],
    stocksDic['initial'][5]['cntNonLedda'], stocksDic['final'][5]['cntNonLedda'], 
    stocksDic['initial'][5]['pctNonLedda'], stocksDic['final'][5]['pctNonLedda'], 
    stocksDic['initial'][5]['meanNonLedda'], stocksDic['final'][5]['meanNonLedda'],
    stocksDic['initial'][5]['sumNonLedda'], stocksDic['final'][5]['sumNonLedda'],  
      
    stocksDic['initial'][6]['cntLedda'], stocksDic['final'][6]['cntLedda'], 
    stocksDic['initial'][6]['pctLedda'], stocksDic['final'][6]['pctLedda'], 
    stocksDic['initial'][6]['meanLedda'], stocksDic['final'][6]['meanLedda'],
    stocksDic['initial'][6]['sumLedda'], stocksDic['final'][6]['sumLedda'],
    stocksDic['initial'][6]['cntNonLedda'], stocksDic['final'][6]['cntNonLedda'], 
    stocksDic['initial'][6]['pctNonLedda'], stocksDic['final'][6]['pctNonLedda'], 
    stocksDic['initial'][6]['meanNonLedda'], stocksDic['final'][6]['meanNonLedda'],
    stocksDic['initial'][6]['sumNonLedda'], stocksDic['final'][6]['sumNonLedda'],  
      
    stocksDic['initial'][7]['cntLedda'], stocksDic['final'][7]['cntLedda'], 
    stocksDic['initial'][7]['pctLedda'], stocksDic['final'][7]['pctLedda'], 
    stocksDic['initial'][7]['meanLedda'], stocksDic['final'][7]['meanLedda'],
    stocksDic['initial'][7]['sumLedda'], stocksDic['final'][7]['sumLedda'],
    stocksDic['initial'][7]['cntNonLedda'], stocksDic['final'][7]['cntNonLedda'], 
    stocksDic['initial'][7]['pctNonLedda'], stocksDic['final'][7]['pctNonLedda'], 
    stocksDic['initial'][7]['meanNonLedda'], stocksDic['final'][7]['meanNonLedda'],
    stocksDic['initial'][7]['sumNonLedda'], stocksDic['final'][7]['sumNonLedda'],  
      
    stocksDic['initial'][8]['cntLedda'], stocksDic['final'][8]['cntLedda'], 
    stocksDic['initial'][8]['pctLedda'], stocksDic['final'][8]['pctLedda'], 
    stocksDic['initial'][8]['meanLedda'], stocksDic['final'][8]['meanLedda'],
    stocksDic['initial'][8]['sumLedda'], stocksDic['final'][8]['sumLedda'],
    stocksDic['initial'][8]['cntNonLedda'], stocksDic['final'][8]['cntNonLedda'], 
    stocksDic['initial'][8]['pctNonLedda'], stocksDic['final'][8]['pctNonLedda'], 
    stocksDic['initial'][8]['meanNonLedda'], stocksDic['final'][8]['meanNonLedda'],
    stocksDic['initial'][8]['sumNonLedda'], stocksDic['final'][8]['sumNonLedda']  
    ))
  
  
  #file1 = open("stocksDic.pickle", "wb") 
  #pickle.dump(stocksDic,file1,2)
  #file1.close()   

  
  return stocksDic


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def make_counts(X, stocksDic):
  """
  Make a dict to hold summary counts and data for initial/final populations
  
  """

  # person groups
  names = {}
  names[0] = 'person_nonmember_NIWF'
  names[1] = 'person_member_NIWF'
  names[2] = 'person_nonmember_unemployed'
  names[3] = 'person_member_unemployed'
  names[4] = 'person_nonmember_NP'
  names[5] = 'person_nonmember_SB'
  names[6] = 'person_member_NP'
  names[7] = 'person_member_SB'
  names[8] = 'person_member_PB'
  
  countsDic = {}
  memberCount = 0
  nonmemberCount = 0
  WF_initial = 0
  unemployed_initial = 0
  WF_final = 0
  unemployed_final = 0
  SB_initial = 0
  SB_final = 0
  NP_initial = 0
  NP_final = 0
  PB_initial = 0
  PB_final = 0
  income_initial = 0
  income_final = 0
  income_final_Ledda = 0
  income_final_nonLedda = 0
  

  for ws in range(9):
    countsDic[names[ws]] = {
      'initial': stocksDic['initial'][ws]['cntLedda'] + stocksDic['initial'][ws]['cntNonLedda'],
      'final'  : stocksDic['final'][ws]['cntLedda'] + stocksDic['final'][ws]['cntNonLedda']
      }
    memberCount += stocksDic['final'][ws]['cntLedda']
    nonmemberCount += stocksDic['final'][ws]['cntNonLedda']
    
    income_initial += stocksDic['initial'][ws]['sumLedda'] + stocksDic['initial'][ws]['sumNonLedda']
    income_final += stocksDic['final'][ws]['sumLedda'] + stocksDic['final'][ws]['sumNonLedda']
    
    income_final_Ledda += stocksDic['final'][ws]['sumLedda']
    income_final_nonLedda += stocksDic['final'][ws]['sumNonLedda']
    
    if ws >= 2:
      WF_initial += stocksDic['initial'][ws]['cntLedda'] + stocksDic['initial'][ws]['cntNonLedda']
      WF_final += stocksDic['final'][ws]['cntLedda'] + stocksDic['final'][ws]['cntNonLedda']
    if ws in [2,3]:
      unemployed_initial += stocksDic['initial'][ws]['cntLedda'] + stocksDic['initial'][ws]['cntNonLedda']
      unemployed_final += stocksDic['final'][ws]['cntLedda'] + stocksDic['final'][ws]['cntNonLedda']

    if ws in [4,6]:
      # nonprofits
      NP_initial += stocksDic['initial'][ws]['cntLedda'] + stocksDic['initial'][ws]['cntNonLedda'] 
      NP_final += stocksDic['final'][ws]['cntLedda'] + stocksDic['final'][ws]['cntNonLedda']
    
    if ws in [5,7]:
      # SB
      SB_initial += stocksDic['initial'][ws]['cntLedda'] + stocksDic['initial'][ws]['cntNonLedda'] 
      SB_final += stocksDic['final'][ws]['cntLedda'] + stocksDic['final'][ws]['cntNonLedda']    

    if ws in [8]:
      # PB
      PB_final += stocksDic['final'][ws]['cntLedda'] + stocksDic['final'][ws]['cntNonLedda']   
     
  countsDic['memberCount'] = memberCount
  countsDic['nonmemberCount'] = nonmemberCount
  countsDic['unemploymemt_rate_initial'] = round(unemployed_initial / WF_initial * 100, 1)
  countsDic['unemploymemt_rate_final'] = round(unemployed_final / WF_final * 100, 1)
  
  countsDic['SB_pct_employed_initial'] = round(SB_initial / (WF_initial - unemployed_initial) *100, 1)
  countsDic['SB_pct_employed_final'] = round(SB_final / (WF_final - unemployed_final) *100, 1)
  
  countsDic['NP_pct_employed_initial'] = round(NP_initial / (WF_initial - unemployed_initial) *100, 1)
  countsDic['NP_pct_employed_final'] = round(NP_final / (WF_final - unemployed_final) *100, 1)

  countsDic['PB_pct_employed_initial'] = round(PB_initial / (WF_initial - unemployed_initial) *100, 1)
  countsDic['PB_pct_employed_final'] = round(PB_final / (WF_final - unemployed_final) *100, 1)  
  
  NP_final_Ledda = stocksDic['final'][6]['cntLedda']
  SB_final_Ledda = stocksDic['final'][7]['cntLedda']
  PB_final_Ledda = stocksDic['final'][8]['cntLedda']
  total_Ledda_employed = NP_final_Ledda + SB_final_Ledda + PB_final_Ledda
  countsDic['NP_pct_employed_Ledda_final'] = round(NP_final_Ledda/total_Ledda_employed * 100, 1)
  countsDic['SB_pct_employed_Ledda_final'] = round(SB_final_Ledda/total_Ledda_employed * 100, 1)
  countsDic['PB_pct_employed_Ledda_final'] = round(PB_final_Ledda/total_Ledda_employed * 100, 1)
  
  countsDic['mean_income_initial'] = int(round(income_initial/X.population))
  countsDic['mean_income_final'] = int(round(income_final/X.population))
  countsDic['mean_income_final_Ledda'] = int(round(income_final_Ledda/memberCount))
  countsDic['mean_income_final_nonLedda'] = int(round(income_final_nonLedda/nonmemberCount))
  
  
  assert np.allclose(memberCount + nonmemberCount, X.population, atol=1)
  assert np.allclose(WF_initial, WF_final, atol=1)
  
  
  return countsDic



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def checkFamilyIncome(X, Year, threshold_family, Membership=None, Idd=None):
  """
  
  Select membership applicants from the population that fall below a specifed threshold for family
  income.  Use income data for Year 0 when selecting applicants.  In a real LEDDA, individuals from 
  families of any income level might join.
  
  """
  
  Wc = X.TF.get_where_list("(R_dollars < " +str(threshold_family)+ ")")
  person1 = X.TF.cols.person1[:][Wc]
  person2 = X.TF.cols.person2[:][Wc]
  persons = np.hstack((person1, person2))
    
  
  W0 = X.TP.get_where_list("pid> -1")  
  inarray = np.in1d(persons, W0)
  persons = persons[inarray==True]
  np.random.shuffle(persons)
  
  if Idd is not None:
    return Idd in persons
  else:
    return persons

 




