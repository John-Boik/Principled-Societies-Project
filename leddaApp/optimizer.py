
import copy, time, pdb

import numpy as np
from scipy.optimize import minimize

from leddaApp import fitness

getFit = fitness.getFit


# ==============================================================================================
np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
np.set_printoptions(6, 120, 150, 220, True)



def gfit(new, x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending):
  
  x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
  
  fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
  fit = fitnessDic['fitness']['total']
  
  if fit < 100:
    fit = 0
  
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
  
  
  TSI_bound = .2
  print("\nBounds:")
  bnds = [[0,1] for x in flexibleList]
  for ii, name in enumerate(flexibleList):
    if "_TS" in name:
      bnds[ii] = (np.maximum(0, X.TSI - X.TSI*TSI_bound), np.minimum(1, X.TSI + X.TSI*TSI_bound))
    bnds[ii] = tuple(bnds[ii])
    print(name, bnds[ii])
  bnds = tuple(bnds)
  
  
  # make initial population
  Pop = []
  flexVals = np.array([X.__dict__[f] for f in flexibleList])
  

  """
  Bounds:
  earmark_NP_donation (0, 1)
  earmark_NP_donation_TS (0.27999999999999997, 0.41999999999999998)
  earmark_PB_subsidy (0, 1)
  earmark_PB_subsidy_TS (0.27999999999999997, 0.41999999999999998)
  earmark_SB_subsidy (0, 1)
  earmark_SB_subsidy_TS (0.27999999999999997, 0.41999999999999998)
  earmark_nurture (0, 1)
  earmark_nurture_TS (0.27999999999999997, 0.41999999999999998)
  person_member_spending_to_member_NP_TS (0.27999999999999997, 0.41999999999999998)
  person_member_spending_to_member_NP_pct (0, 1)
  person_member_spending_to_member_PB_TS (0.27999999999999997, 0.41999999999999998)
  person_member_spending_to_member_PB_pct (0, 1)
  person_member_spending_to_member_SB_TS (0.27999999999999997, 0.41999999999999998)
  person_member_spending_to_member_SB_pct (0, 1)
  person_member_spending_to_nonmember_NP_TS (0.27999999999999997, 0.41999999999999998)
  person_member_spending_to_nonmember_NP_pct (0, 1)
  person_member_spending_to_nonmember_SB_TS (0.27999999999999997, 0.41999999999999998)
  person_member_spending_to_nonmember_SB_pct (0, 1)
  person_nonmember_spending_to_member_NP_pct (0, 1)
  person_nonmember_spending_to_member_PB_pct (0, 1)
  person_nonmember_spending_to_member_SB_pct (0, 1)
  person_nonmember_spending_to_nonmember_NP_pct (0, 1)
  person_nonmember_spending_to_nonmember_SB_pct (0, 1)

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
    x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
    
    # get fitness
    fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
    x.Fitness = fitnessDic['fitness']['total']
    #print("  fitness= ", x.Fitness, "\n")
    Pop.append(x)
    
  
  scores = np.array([x.Fitness for x in Pop])
  idd = np.argsort(scores)
  scores = scores[idd]
  Pop = [Pop[i] for i in idd]
  
  xinitial = copy.deepcopy(Pop[0])
  
  print("\ninitial best fit = ", xinitial.Fitness)
  print("\ninitial best new = ", xinitial.new) 
  test1 = np.array([xinitial.__dict__[f] for f in flexibleList])
  assert np.allclose(test1, xinitial.new)  
  
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
    newPop = [copy.deepcopy(Pop[0])]
      
    # make new population
    for n in range(popSize):
      x = copy.deepcopy(X)
      
      # get random parents from top 50
      i0, i1 = np.random.permutation(range(1, 50))[0:2] # top 50
      #if i%5== 0:
      v0 = Pop[0].new.copy()
      #else:
      #  p0 = Pop[0]
      v1 = Pop[i1].new.copy()
      
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
      
      x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
      
      test1 = np.array([x.__dict__[f] for f in flexibleList])
      assert np.allclose(test1, x.new)
    
      # get fitness
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      x.Fitness = fitnessDic['fitness']['total']
      newPop.append(x)

      test2 = np.array([x.__dict__[f] for f in flexibleList])
      assert np.allclose(test2, x.new)
    
    Pop = newPop[0:popSize]
    scores = np.array([x.Fitness for x in Pop])
    idd = np.argsort(scores)
    scores = scores[idd]
    Pop = [Pop[i] for i in idd]
    
    if 1==1:
      # call optimize
      t0 = time.time()
      x = Pop[0]
      
      print("\nx.new prior: ", x.new)
      test1 = np.array([x.__dict__[f] for f in flexibleList])
      assert np.allclose(test1, x.new)
      
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      print ("\nprior fit: ", fitnessDic['fitness']['total'])
      
      res = minimize(gfit, x.new, 
        args=(x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending), 
        method='L-BFGS-B',
        bounds = bnds, 
        options= {
          'disp': True,
          'ftol' : 1e13 * np.finfo(float).eps,
          'maxls':200
          })
      
      new = res['x']
      
      print("success: ", res['success'])
      
      x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
      x.Fitness = res['fun']
      
      
      
      print('\nbfgs: ', x.Fitness, x.new, "\n")
      
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      print ("post fit: ", fitnessDic['fitness']['total'])
      
      raise Exception()
      
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
  if not isinstance(new, np.ndarray):
    new = np.array(new)
  
  # put unadjusted new into x
  _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
  
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

  if np.any(np.isnan(new)):
    pdb.set_trace()
  
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
 
  if np.any(np.isnan(new)):
    pdb.set_trace()
    
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
  
  if len(idx) > 0:
    if unfixed == 0:
      # set all spending to an equal number
      new[idx] = 1 / len(new[idx]) 
    else:
      mult = (1 - fixed) / unfixed
      new[idx] = new[idx] * mult 

  if np.any(np.isnan(new)):
    pdb.set_trace()

  # each member spending_TS must be < .999
  idx = []
  for name in memberSpending:
    if name[0:-3] + "_TS" in flexibleList:
      idx.append(flexibleList.index(name[0:-3] + "_TS"))
  new[idx] = np.minimum(new[idx], .999)
  
  if np.any(np.isnan(new)):
    pdb.set_trace()

  # put adjusted new into x
  _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
  x.new = new
  
  return x


  
