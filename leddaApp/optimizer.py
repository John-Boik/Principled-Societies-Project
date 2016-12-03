
import copy, time, pdb, re

import numpy as np
from scipy.optimize import minimize

from leddaApp import fitness

getFit = fitness.getFit


# ==============================================================================================
np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
np.set_printoptions(6, 120, 150, 220, True)



def gfit(new, x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending):
  """
  This is the function that calls getFit for the the BFGS optimization
  """
  
  x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
  
  fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=True, Optimize=True)
  fit = fitnessDic['fitness']['total']
  
  #if fit < 0:
  #  fit = 0
  
  return fit



# ==============================================================================================
# Main function
# ==============================================================================================

def genetic(X, stocksDic):
  """
  Setup the optimizer (genetic algorithm and BFGS for now), then call the fitness function repeatedly.
  If BFGS, do the optimization and return.  If both, run the genetic algorithm and at the end of each 
  generation run BFGS with mild restrictions on function calls. Run BFGS again at the end with no 
  restrictions on function calls.
  
  In any case, return when the fitness is <= 100 (to save computation load)
  """
  
  # names of potential flexible variables for ensuring constraints
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
  
  
  maxGenerations = 200
  popSize = 100

  flexibleList = X.flexibleList
  
  print("\nBounds:")
  bounds = [[0,1] for x in flexibleList]
  for ii, name in enumerate(flexibleList):
    if re.match("earmark_[A-Za-z_]+_TS", name):
      bounds[ii] = [X.earmarks_TS_lb, X.earmarks_TS_ub]
    if re.match("person_member_spending_to_member_[A-Z]+_TS", name):
      bounds[ii] = [X.spending_TS_lb, X.spending_TS_ub]    
    bounds[ii] = tuple(bounds[ii])
    print("{:<50s} {:>.4f}, {:>.4f}".format(name, bounds[ii][0], bounds[ii][1]))
  bounds = tuple(bounds)
  
  
  # initial values for flexible variables
  if X.doRandomStart:
    # user random values
    flexVars = np.random.uniform(size=len(flexibleList))
  else:
    # use user-supplied values
    flexVars = np.array([X.__dict__[f] for f in flexibleList])
  
  # adjust flexVars as needed and put back into X
  X = adjust_flexible_values(X, flexVars, earmarks, nonmemberSpending, memberSpending)  
  flexVars = X.new
  
  
  if not X.doGenetic:
    # only do BFGS optimization, then return
    x = copy.deepcopy(X)
    res = minimize(gfit, x.new, 
      args=(x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending), 
      method='L-BFGS-B',
      bounds = bounds, 
      options= {
        'disp': False,
        'ftol' : 1e12 * np.finfo(float).eps,
        'maxls':200,
        })    

    new = res['x']
    print("success: ", res['success'])
    x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
    x.Fitness = res['fun']
    print('\nbfgs fitness: ', x.Fitness, "\n")
    
    fitnessDic, tableDic, summaryGraphDic = fitness.getFit(x, stocksDic, Print=True, Optimize=False)
    print ("fit: ", fitnessDic['fitness']['total'])  

    return fitnessDic, tableDic, summaryGraphDic   


  # ----------------------------------------------------------------------
  # Genetic algorithm
  # ----------------------------------------------------------------------  
  
  # create an intial population
  Pop = []  
  for n in range(popSize):
    x = copy.deepcopy(X)
    new = flexVars.copy()
    if n != 0:
      # for first in population, just use initial variables
      # choose a variance, choose which elements to replace, and make a new vector
      variance  = np.random.uniform(1e-8, X.Variance)
      Size = np.random.randint(1, len(new)+1, 1)
      ia = np.sort(np.random.permutation(len(new))[0:Size])
      new[np.ix_(ia)] = abs(np.random.normal(new[np.ix_(ia)], variance))
    
    # adjust new as needed and put back into x
    x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
    
    # get fitness
    fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
    x.Fitness = fitnessDic['fitness']['total']
    #print("  fitness= ", x.Fitness)
    Pop.append(x)
    
  
  scores = np.array([x.Fitness for x in Pop])
  idd = np.argsort(scores)
  scores = scores[idd]
  Pop = [Pop[i] for i in idd]
  
  xinitial = copy.deepcopy(Pop[0])
  
  print("\ninitial best fit = ", xinitial.Fitness)
  #print("\ninitial best new = ", xinitial.new) 
  test1 = np.array([xinitial.__dict__[f] for f in flexibleList])
  assert np.allclose(test1, xinitial.new)  
  
  print("\nBest initial score is: {:,}".format(scores[0]))
  

  
  # ================================================================ 
  # Do generations
  # ================================================================ 
  
  Result = []
  for gen in range(maxGenerations):
    
    print (" ------------ gen = {:3d} --------- fit= {:20,.0f}---".format(gen, Pop[0].Fitness))
    
    # retain the best gene from the previous population
    #if (gen>0) and (gen%2 == 0):
    #  newPop = [copy.deepcopy(Pop[40])]
    #else:
    #  newPop = [copy.deepcopy(Pop[0])]
    newPop = []
      
    # make new population
    for n in range(popSize):
      x = copy.deepcopy(X)
      
      # get random parents from top 50
      i0, i1 = np.random.permutation(range(1, 50))[0:2] # top 50
      if gen%2 == 0:
        v0 = Pop[i0].new.copy()
      else:
        v0 = Pop[0].new.copy()
      v1 = Pop[i1].new.copy()
      
      # make random vector using: new = (v1 -v0) * random + v0
      variance  = np.random.uniform(1e-8, X.Variance)
      ran = np.random.normal(0,variance, len(v0))
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
    
    if X.doBFGS:
      # call optimize
      x = copy.deepcopy(Pop[0])
      
      test1 = np.array([x.__dict__[f] for f in flexibleList])
      assert np.allclose(test1, x.new)
      
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      
      res = minimize(gfit, x.new, 
        args=(x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending), 
        method='L-BFGS-B',
        bounds = bounds, 
        options= {
          'disp': False,
          'ftol' : 1e11 * np.finfo(float).eps,
          'maxls':200,
          'maxiter':2
          })
      
      new = res['x']
      x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
      x.Fitness = res['fun']
      
      print('bfgs: ', x.Fitness, "success: ", res['success'], "\n")
      
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      
      if x.Fitness < scores[0]:
        Pop[0] = x
        scores[0] = x.Fitness 
    
    bestf = scores[0]
    bestx = Pop[0]
    
    Result.append(bestx)
    
    #print("\nGen= ", gen, "score= {:,}".format(scores[0]))
    
    if bestf <= 1:
      break
    
    if (len(Result) > 10)  and (np.unique([int(round(x.Fitness)) for x in Result[-10:]]).size == 1):
      break
  
  # ==================================================================
  Pop = Result
  scores = np.array([x.Fitness for x in Pop])
  idd = np.argsort(scores)
  scores = scores[idd]
  Pop = [Pop[i] for i in idd]
  print("\nfinal fit genetic: ", Pop[0].Fitness)
  
  if X.doBFGS:
    # call optimize
    t0 = time.time()
    x = copy.deepcopy(Pop[0])
    new = x.new
    res = minimize(gfit, x.new, 
      args=(x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending), 
      method='L-BFGS-B',
      bounds = bounds, 
      options= {
        'disp': False,
        'ftol' : 1e11 * np.finfo(float).eps,
        })

    new = res['x']
    x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
    x.Fitness = res['fun']
    print('final bfgs: ', x.Fitness, "success: ", res['success'], "\n")
    if  x.Fitness < scores[0]:
      scores[0] = x.Fitness
      Pop[0] = x  
            
  new = Pop[0].new
  x = copy.deepcopy(Pop[0])
  #x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending)
  
  # run fitness for the winning gene, printing results  
  fitnessDic, tableDic, summaryGraphDic = fitness.getFit(x, stocksDic, Print=True, Optimize=False)

  print('final genetic: ', x.Fitness, " fitdic: ", fitnessDic['fitness']['total'], "\n")
  flexVars = np.array([x.__dict__[f] for f in flexibleList])
  
  print("\nFlexible:")
  for ii, f in enumerate(flexibleList):
    print("['{:}', {:}],".format(f, flexVars[ii]*100))
  
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


  
