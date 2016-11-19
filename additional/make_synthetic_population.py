
import numpy as np
import tables as tb
from scipy import stats
import sys, os, pickle, shutil, datetime

sys.path.append(os.path.join('..', 'leddaApp'))
import Config


"""
This script creates and saves a synthetic population, based on Census microdata for Lane
County Oregon. It includes person and family incomes. The process is described in 

Boik, J. (2014) ‘First Micro-Simulation Model of a LEDDA Community Currency-Dollar 
Economy’ International Journal of Community Currency Research 18 (A) 11-29 
<www.ijccr.net> ISSN 1325-9547 http://dx.doi.org/10.15133/j.ijccr.2014.002. 

The file INCTOT_2011_1yr.pickle contains income data from Lane County 2011 US Census microdata. It
is sampled here to create synthetic populations. The process is described in the above paper.

Once the 'steady_state_population.hdf5' file is created, it is copied to the 
'../leddaApp/static/data' folder. Contents there are overwritten.

To run:
  at python interactive cursor:
    exec(open('./make_synthetic_population.py').read()), or 
  in command window:
    python make_synthetic_population.py

"""


#########################################################################################
# Logger class
#########################################################################################
class Logger(object):
    def __init__(self, Folder):
        self.terminal = sys.stdout
        
        fn = os.path.join(Folder, "make_synthetic_Population.log")
        self.log = open(fn, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def close(self):
      self.log.flush()
      self.log.close()

    def flush(self):
      self.log.flush()      


#########################################################################################
# Setup Function
#########################################################################################

def setup():
  """
  Setup some parameters for creating the HDF5 file. Use the imported Config module as a container.
  """
  
  # the HDF5 file will contain numerous populations, one each for numerous starting unemployment
  # levels. Note that fine divisions here would require large population sizes to be meaningful,
  # so only a few example initial unemployment rates are used.
  Config.rate_unempoyment_initial = np.array([3., 5., 7., 11., 15., 20., 25., 30.]) 
  

  Config.number_employed = np.round(Config.population * \
    Config.labor_participation_rate/100 * (1-Config.rate_unempoyment_initial/100), 0).astype('i')
  
  Config.number_unemployed = np.round(Config.population * \
    Config.labor_participation_rate/100 * Config.rate_unempoyment_initial/100, 0).astype('i')
  
  Config.number_NIWF = np.round(Config.population * \
    (1-Config.labor_participation_rate/100), 0).astype('i')
  
  # adjust if rounding error
  for i in range(Config.rate_unempoyment_initial.size):
    if Config.population - (Config.number_employed[i] + 
      Config.number_unemployed[i] + Config.number_NIWF) == 1:
        Config.number_unemployed[i] += 1
    if Config.population - (Config.number_employed[i] + 
      Config.number_unemployed[i] + Config.number_NIWF) == -1:
        Config.number_unemployed[i] -= 1     
  
    assert np.allclose(Config.population, Config.number_employed[i] + \
      Config.number_unemployed[i] + Config.number_NIWF)

  # create new pytables table
  makeHDF5()




#########################################################################################
# Make the HDF5 file
#########################################################################################
def makeHDF5():
  """
  
  Create a new HDF5 file.  The HDF5 contains the person table (TP) and family 
  table (TF), one each for every Config.rate_unempoyment_initial.  
  
  Column names are mostly self-explanatory.  See abbreviations list for assistance.  A few 
  column names are listed below:

    pid or fid            = person or family unique ID number
    R_*                   = prefix for received
    P_*                   = prefix for paid                    
    P_donation_NP_dollars = dollar donations paid to nonprofits apart from CBFS contributions
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
  
  hdf5_filename = 'steady_state_population.hdf5'
  
  
  # open income files obtained from Census microdata for Lane County, OR 
  file1 = open("INCTOT_2011_1yr.pickle", "rb")  
  INCTOT = pickle.load(file1, encoding='latin1')
  file1.close()  

  # use only positive incomes 
  INCTOT = INCTOT[INCTOT>0]
  
  HDF5 = tb.openFile(hdf5_filename, mode='w')
  
  # =====================================================================
  # Persons Table
  # =====================================================================
  
  DescriptionP = {}
  for ivar, Var in enumerate([
    'pid', 'fid', 'work_status', 'partner',
        
    'R_wages_SB_dollars', 'R_wages_NP_dollars',
    'R_gov_support_dollars',

    'P_gov_tax_dollars', 
    'P_donation_NP_dollars',
    
    'R_dollars', 
    'P_dollars', 
    ]):      
    
    if Var in ['membership', 'pid', 'work_status', 'partner', 'fid']:
      DescriptionP[Var] = tb.Int64Col(dflt=0,pos=ivar)       
    else:
      DescriptionP[Var] = tb.Float64Col(dflt=0,pos=ivar)   
  
  
  # iterate over unemployment rates 
  sys.stdout.flush()
  for ii, unemPct in enumerate(Config.rate_unempoyment_initial):
    TP = HDF5.createTable('/','Persons_' + str(ii), description=DescriptionP, 
    title='Table of individuals', \
      expectedrows= Config.population)
    
    TP.attrs.unemployment_rate = unemPct
    # choose hi/low ranges for selecting incomes for employed vs unemployed & NIWF persons. Break is 
    # chosen so that the mean and median are roughly equal to those published for Lane County, OR,
    # for the 11 percent unemployment in 2011
    Cut = Config.employed_income_Census_threshold
    INC_W0 = np.where(INCTOT < Cut)[0]    # incomes for unemployed NIWF people
    INC_W1 = np.where(INCTOT >= Cut)[0]   # incomes for employed people
    
    # incomes for employees of nonprofits. A threshold is placed on upper amount of earnings to
    # prevent the unnatural situation where a ultra-high earner comes from the nonprofit sector
    upperNP = Config.nonprofit_income_Census_threshold
    INC_NP = np.where((INCTOT >= Cut) & (INCTOT <= upperNP))[0] 
        
    # fill table for employed persons (SB and NP)
    row = TP.row
    for i in range(0, Config.number_employed[ii]):
      row['pid'] = i
      row['work_status'] = np.random.choice([4,5], 1, \
        p = [Config.WF_pct_NP_initial/100, 1-Config.WF_pct_NP_initial/100])
      if row['work_status'] == 4:
        # nonprofits
        income = np.random.choice(INCTOT[INC_NP], size=1, replace=True)  
        row['R_wages_NP_dollars'] = income  
      else:
        # forprofits
        income = np.random.choice(INCTOT[INC_W1], size=1, replace=True)  
        row['R_wages_SB_dollars'] = income     
      row['R_dollars'] = income
      row.append()
    TP.flush()

    # fill table for unemployed persons
    for j in range(i+1, i+1+Config.number_unemployed[ii]):
      row['pid'] = j
      row['work_status'] = 2
      row['R_gov_support_dollars'] = np.random.choice(INCTOT[INC_W0], size=1, replace=True)                 
      row['R_dollars'] = row['R_gov_support_dollars']
      row.append()
    TP.flush()

    # fill table for not in work force (NIWF) persons
    for k in range(j+1, j+1+Config.number_NIWF):
      row['pid'] = k
      row['work_status'] = 0
      row['R_gov_support_dollars'] = np.random.choice(INCTOT[INC_W0], size=1, replace=True)                 
      row['R_dollars'] = row['R_gov_support_dollars']
      row.append()
    TP.flush()
    
    incomeP = TP.cols.R_wages_NP_dollars[:] + TP.cols.R_wages_SB_dollars[:] + \
      TP.cols.R_gov_support_dollars[:]
    
    assert np.allclose(incomeP, TP.cols.R_dollars[:])
    assert np.all(incomeP > 0)
    
    print("\n\n===========\nii= {:d}, unemployment rate= {:f}".format(ii, unemPct))
    print("\ncounty income, mean = ${0:,.9g}".format(np.round(incomeP.mean(),0).item()))
    print("county income, median = ${0:,.9g}\n".format(np.round(np.median(incomeP),0).item()))


    # =====================================================================
    # Families Table
    # =====================================================================
    
    DescriptionF = {}
    for ivar, Var in enumerate(['fid', 'person1', 'person2', 
      'R_dollars']):
                 
      if Var in ['fid', 'person1', 'person2']:
        DescriptionF[Var] = tb.Int64Col(dflt=0,pos=ivar)       
      else:
        DescriptionF[Var] = tb.Float64Col(dflt=0,pos=ivar)   
    
    TF = HDF5.createTable('/','Families_' + str(ii), description=DescriptionF, 
    title='Table of families', \
      expectedrows= Config.population/2)
    
    TF.attrs.unemployment_rate = unemPct
    
    # generate random pairings of 2 people per family
    ran0 = np.random.permutation(Config.population)
    ran1 = ran0[0:ran0.size/2]
    ran2 = ran0[ran0.size/2:]

    # enter base info for Families table ------------------------------------------
    row = TF.row
    for i in range(0, int(Config.population/2)): 
      row['fid'] = i
      row['person1'] = ran1[i]
      row['person2'] = ran2[i]
      
      income1 = TP[ran1[i]]['R_dollars'] 
      income2 = TP[ran2[i]]['R_dollars']         

      row['R_dollars'] = income1 + income2      
      row.append()      
      
      # set fid and partners in Table Persons
      TP.cols.fid[ran1[i]] = i
      TP.cols.fid[ran2[i]] = i
      TP.cols.partner[ran1[i]] = ran2[i]
      TP.cols.partner[ran2[i]] = ran1[i]

    TF.flush()
    
    # create indexes for faster searching
    TP.cols.pid.create_index()
    TP.cols.fid.create_index()
    TP.cols.work_status.create_index()
    TP.cols.partner.create_index()
    TF.cols.fid.create_index()
    
    TP.filters.complevel = 5
    TF.filters.complevel = 5 
    
    # print results -------------------------------------------------------------------
    print("\nfamily income, mean = ${0:,.9g}".format(
      np.round(TF.cols.R_dollars[:].mean(),0).item()))
    
    print("family income, median = ${0:,.9g}\n".format(
      np.round(np.median(TF.cols.R_dollars[:]),0).item()))
    
    for i in np.linspace(0,100,21):
      print("  percentile = {0:>3.0f},  person income = ${1:>12,.9g}".format(
        i, np.round(stats.scoreatpercentile(incomeP, i),0).item()))
    print("\n")
    
    for i in np.linspace(0,100,21):
      print("  percentile = {0:>3.0f},  family income = ${1:>12,.9g}".format(
        i, np.round(stats.scoreatpercentile(TF.cols.R_dollars[:], i),0).item()))
    print("\n")

    # save income distributions
    Max = TF.cols.R_dollars[:].max()
    
    #Max = int(np.ceil(Max/1000)*1000)
    bins = np.arange(0,Max+5000,5000)
    print("len bins= ", bins.size)
    print("\nMax = {:,}, max bin={:,}".format(Max, bins[-1]))
    
    hist, bin_edges = np.histogram(TF.cols.R_dollars[:], bins=bins, density=False)
    hist = hist.tolist()
    bin_edges = bin_edges.tolist()
    TF.attrs.hist = hist
    TF.attrs.bin_edges = bin_edges
    
  
  
  HDF5.close() 
  
  
  shutil.copy(hdf5_filename, '../leddaApp/static/data/')
  


#########################################################################################
# Run script
#########################################################################################

if __name__ == "__main__":
  # log output
  sys.stdout = Logger('./')

  tm = datetime.datetime.today().strftime('%Y-%b-%d at %H:%M')
  print ("""
*****************************************************************
Make Synthetic Population
date   = {:s}
*****************************************************************
    """.format(tm))

  # print out Config parameters
  print("\nConfig Parameters:")  
  keys = dir(Config)
  for key in keys:
    if key[0] == "_":
      continue
    
    value = Config.__dict__[key]
    print("  {}: {}".format(key, value))
  print("\n")  
  
  sys.stdout.flush()
  setup()



