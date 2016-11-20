
import copy, time

import numpy as np
from scipy.optimize import minimize

from leddaApp import fitness

getFit = fitness.getFit


# ==============================================================================================
np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
np.set_printoptions(6, 120, 150, 220, True)



def gfit(new, x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending):
  
  new = np.exp(new)
  _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
  
  #print("new2: ", new)
  
  # adjust new as needed and put back into x
  new = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
  _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
  
  #print("new3: ", new)
  
  fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
  fit = fitnessDic['fitness']['total']
  return fit



# ==============================================================================================
# Main function
# ==============================================================================================

def genetic(X, stocksDic):
  """
  Setup the optimizer (genetic algorithm, for now), then call the fitness function repeatedly
  """
  earmarks = ['earmark_NP_donation', 'earmark_nurture', 'earmark_PB_subsidy', 'earmark_SB_subsidy']
  nonmemberSpending = ['person_nonmember_spending_to_member_NP_pct', 
    'person_nonmember_spending_to_member_PB_pct', 
    'person_nonmember_spending_to_member_SB_pct',
    'person_nonmember_spending_to_nonmember_NP_pct',
    'person_nonmember_spending_to_nonmember_SB_pct']
  
  memberSpending = ['person_member_spending_to_member_NP_pct',
    'person_member_spending_to_member_PB_pct',
    'person_member_spending_to_member_SB_pct',
    'person_member_spending_to_nonmember_NP_pct',
    'person_member_spending_to_nonmember_SB_pct']
  
  
  maxGenerations = 100
  popSize = 100

  flexibleList = X.flexibleList
  
  # make initial population
  Pop = []
  flexVals = np.array([X.__dict__[f] for f in flexibleList])
  
  print("\nFlexible:")
  for ii, f in enumerate(flexibleList):
    print(ii, f, "  ", flexVals[ii])

  """
  Flexible:
  0 earmark_NP_donation    0.08
  1 earmark_NP_donation_TS    0.35
  2 earmark_PB_subsidy    0.1
  3 earmark_PB_subsidy_TS    0.35
  4 earmark_SB_subsidy    0.06
  5 earmark_SB_subsidy_TS    0.35
  6 earmark_nurture    0.37
  7 earmark_nurture_TS    0.35
  8 person_member_spending_to_member_NP_TS    0.35
  9 person_member_spending_to_member_NP_pct    0.2
  10 person_member_spending_to_member_PB_TS    0.35
  11 person_member_spending_to_member_PB_pct    0.2
  12 person_member_spending_to_member_SB_TS    0.35
  13 person_member_spending_to_member_SB_pct    0.2
  14 person_member_spending_to_nonmember_NP_TS    0.35
  15 person_member_spending_to_nonmember_NP_pct    0.2
  16 person_member_spending_to_nonmember_SB_TS    0.35
  17 person_member_spending_to_nonmember_SB_pct    0.2
  18 person_nonmember_spending_to_member_NP_pct    0.2
  19 person_nonmember_spending_to_member_PB_pct    0.2
  20 person_nonmember_spending_to_member_SB_pct    0.2
  21 person_nonmember_spending_to_nonmember_NP_pct    0.2
  22 person_nonmember_spending_to_nonmember_SB_pct    0.2
  """
  
  for n in range(popSize):
    x = copy.deepcopy(X)
    if n == 0:
      # use original
      new = flexVals.copy()
    else:
      # make new random version
      new = abs(np.random.normal(flexVals, x.Variance))
    _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
    
    # adjust new as needed and put back into x
    new = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
    _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
    x.new = new
    
    # get fitness
    fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
    x.Fitness = fitnessDic['fitness']['total']
    #print("  fitness= ", x.Fitness, "\n")
    Pop.append(x)
    
  
  scores = np.array([x.Fitness for x in Pop])
  idd = np.argsort(scores)
  scores = scores[idd]
  Pop = [Pop[i] for i in idd]
  
  print("\nBest initial score is: {:,}".format(scores[0]))
  

  
  # ================================================================ 
  # Do generations
  # ================================================================ 
  
  Result = []
  for gen in range(maxGenerations):
    
    print ("\n ------------ gen = {:d} ------------".format(gen))
    
    # retain the best gene from the previous population
    #if (gen>0) and (gen%4 == 0):
    #  newPop = [Pop[40]]
    #else:
    newPop = [Pop[0]]
      
    # make new population
    for n in range(popSize):
      x = copy.deepcopy(X)
      
      # get random parents from top 50
      i0, i1 = np.random.permutation(range(1, 50))[0:2] # top 50
      #if i%5== 0:
      p0 = Pop[0]
      #else:
      #  p0 = Pop[0]
      p1 = Pop[i1]
      
      # get variables
      v0 = p0.new
      v1 = p1.new
      
      # make random vector using: new = (v1 -v0) * random + v0
      ran = np.random.normal(0,.3, len(v0))
      Size = np.random.randint(1, len(v0)+1, 1)
      ia = np.sort(np.random.permutation(len(v0))[0:Size])
      new = v0
      new[np.ix_(ia)] = (v1[np.ix_(ia)]-v0[np.ix_(ia)]) * ran[np.ix_(ia)] + v0[np.ix_(ia)]
      new = np.abs(new)
      
      # do a mutation every now and then
      if np.random.rand() < .2:
        tmp = np.random.uniform(0,1, len(v0))
        Size = np.random.randint(1, len(v0)+1, 1)
        ia = np.sort(np.random.permutation(len(v0))[0:Size])
        new[np.ix_(ia)] = tmp[np.ix_(ia)]
   
      # adjust new as needed and put back into x
      _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
      new = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
      _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
      x.new = new

    
      # get fitness
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      x.Fitness = fitnessDic['fitness']['total']
      newPop.append(x)
    
    Pop = newPop[0:popSize]
    scores = np.array([x.Fitness for x in Pop])
    idd = np.argsort(scores)
    scores = scores[idd]
    Pop = [Pop[i] for i in idd]
    
    if 1==2:
      # call optimize
      t0 = time.time()
      x = Pop[0]
      res = minimize(gfit, np.log(x.new), args=(x, stocksDic, flexibleList, earmarks, 
        nonmemberSpending, memberSpending), method='L-BFGS-B', 
        options={'disp': True, 'maxiter':1000000, 'maxfun':50, 'maxls':200, 
        'ftol' : 1e12 * np.finfo(float).eps})
      new = np.exp(res['x'])
      new = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
      _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
      x.new = new
      x.Fitness = res['fun']
      Pop[0] = x
      scores[0] = x.Fitness 
    
    bestf = scores[0]
    bestx = Pop[0]
    
    Result.append(bestx)
    
    print("\nGen= ", gen, "score= {:,}".format(scores[0]))
    
    if bestf <= 1:
      break
    
    if (len(Result) > 4)  and (Result[-4].Fitness == Result[-1].Fitness):
      break
  
  # ==================================================================
  Pop = Result
  scores = np.array([x.Fitness for x in Pop])
  idd = np.argsort(scores)
  scores = scores[idd]
  Pop = [Pop[i] for i in idd]
  bestx = Pop[0]
  print(Pop[0].Fitness)
 

  try:
    raise Exception()
    # call optimize
    t0 = time.time()
    x = Pop[0]
    new = x.new
    print("new: ", x.Fitness)
    res = minimize(gfit, np.log(new), args=(x, stocksDic, flexibleList, earmarks, 
      nonmemberSpending, memberSpending), method='L-BFGS-B', 
      options={'disp': True, 'maxiter':1000000, 'maxfun':10000, 'maxls':2000, 
      'ftol' : 1e12 * np.finfo(float).eps})
    
    new = np.exp(res['x'])
    new = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
    _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
    x.new = new
    x.Fitness = res['fun']
        
    Pop[0] = x
    scores[0] = x.Fitness  
    bestx = Pop[0]
    
  except: 
    pass



  # run fitness for the winning gene, printing results  
  fitnessDic, tableDic, summaryGraphDic = fitness.getFit(bestx, stocksDic, Print=False, Optimize=False)

  flexVals = np.array([bestx.__dict__[f] for f in flexibleList])
  
  print("\nFlexible:")
  for ii, f in enumerate(flexibleList):
    print(ii, f, "  ", flexVals[ii])
  
  return fitnessDic, tableDic, summaryGraphDic    




#######################################################################################
# helper functions
#######################################################################################


def adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending):
  """
  Adjusts any values for flexible so that constraints are met
  """
  
  flexibleList = x.flexibleList
  
  # All earmarks must sum to < .999
  fixed = 0
  unfixed = 0
  idx = []
  for name in earmarks:
    if name in flexibleList:
      idx.append(flexibleList.index(name))
      unfixed += x.__dict__[name]
    else:
      fixed += x.__dict__[name]

  if fixed + unfixed > .999:
    mult = (.999 - fixed) / unfixed
    new[idx] = new[idx] * mult  
  
  # each earmark_TS must be TSI < ts < .999
  idx = []
  for name in earmarks:
    if name + "_TS" in flexibleList:
      idx.append(flexibleList.index(name + "_TS"))
  new[idx] = np.minimum(new[idx], .999 )  
  new[idx] = np.maximum(new[idx], x.TSI)  

  # total fraction nonmember spending must be == 1
  fixed = 0
  unfixed = 0
  idx = []
  for name in nonmemberSpending:
    if name in flexibleList:
      idx.append(flexibleList.index(name))
      unfixed += x.__dict__[name]
    else:
      fixed += x.__dict__[name]
  mult = (1 - fixed) / unfixed
  new[idx] = new[idx] * mult 
 

  # total fraction member spending must be == 1
  fixed = 0
  unfixed = 0
  idx = []
  for name in memberSpending:
    if name in flexibleList:
      idx.append(flexibleList.index(name))
      unfixed += x.__dict__[name]
    else:
      fixed += x.__dict__[name]
  mult = (1 - fixed) / unfixed
  new[idx] = new[idx] * mult 

  # each member spending_TS must be < .999
  idx = []
  for name in memberSpending:
    if name[0:-3] + "_TS" in flexibleList:
      idx.append(flexibleList.index(name[0:-3] + "_TS"))
  new[idx] = np.minimum(new[idx], .999)


  return new


  
