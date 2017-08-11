
import copy, time, pdb, re

import numpy as np
from scipy.optimize import minimize

from leddaApp import fitness

getFit = fitness.getFit


# ==============================================================================================
np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
np.set_printoptions(6, 120, 150, 220, True)



def gfit(new, x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending, bounds):
  """
  This is the function that calls getFit for the the BFGS optimization
  """
  
  x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending, bounds)
  
  fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=True, Optimize=True)
  fit = fitnessDic['fitness']['total']
  
  if fit < 1000:
    fit = 0
  
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
  
  nonmemberSpending = ['person_nonmember_spending_to_member_NP', 
    'person_nonmember_spending_to_member_PB', 
    'person_nonmember_spending_to_member_SB',
    'person_nonmember_spending_to_nonmember_NP',
    'person_nonmember_spending_to_nonmember_SB']
  
  memberSpending = ['person_member_spending_to_member_NP',
    'person_member_spending_to_member_PB',
    'person_member_spending_to_member_SB',
    'person_member_spending_to_nonmember_NP',
    'person_member_spending_to_nonmember_SB']
  

  maxGenerations = 200
  popSize = 100

  flexibleList = X.flexibleList
  
  print("\nBounds:")
  bounds = [[0,1] for x in flexibleList]
  for ii, name in enumerate(flexibleList):
    if re.match("earmark_[A-Za-z_]+_TS", name):
      bounds[ii] = [X.earmarks_TS_lb, X.earmarks_TS_ub]
    
    elif re.match("person_member_spending_to_member_[A-Z]+_TS", name):
      bounds[ii] = [X.spending_TS_lb, X.spending_TS_ub]    

    elif '_pct' in name:
      bounds[ii] = [X.spending_pct_lb, X.spending_pct_ub]       

    
    bounds[ii] = tuple(bounds[ii])
    print("{:<50s} {:>.4f}, {:>.4f}".format(name, bounds[ii][0], bounds[ii][1]))
  bounds = tuple(bounds)
  print("\n\n")
  
  
  # initial values for flexible variables
  if X.doRandomStart:
    # user random values
    flexVars = np.random.uniform(size=len(flexibleList))
  else:
    # use user-supplied values
    flexVars = np.array([X.__dict__[f] for f in flexibleList])
  
  
  # adjust flexVars as needed and put back into X
  X = adjust_flexible_values(X, flexVars, earmarks, nonmemberSpending, memberSpending, bounds)  
  flexVars = X.new
  
  if not X.doGenetic:
    # only do BFGS optimization, then return
    x = copy.deepcopy(X)
    res = minimize(gfit, x.new, 
      args=(x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending, bounds), 
      method='L-BFGS-B',
      bounds = bounds, 
      options= {
          'disp': True,
          'ftol' : 1e-15, #* np.finfo(float).eps,  #high is weak, fast
          'maxls':200,
          'maxiter':200
        })    

    new = res['x']
    x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending, bounds)
    x.Fitness = res['fun']
    print('\nbfgs: {:,}, success: {}, Msg: {}\n'.format(x.Fitness, res['success'], res['message']))
    
    fitnessDic, tableDic, summaryGraphDic = fitness.getFit(x, stocksDic, Print=True, Optimize=False)
    print ("fit: ", fitnessDic['fitness']['total'])  

    flexVars = np.array([x.__dict__[f] for f in flexibleList])
    flexDic = {}
      
    for ii, f in enumerate(flexibleList):
      flexDic[f] = [flexVars[ii]*100]
      print("'{:}': [{:}],".format(f, flexVars[ii]*100))
    
    fitnessDic['flexDic'] = flexDic

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
      Size = np.random.randint(1, len(new)+1, 1).item()
      ia = np.sort(np.random.permutation(len(new))[0:Size])
      new[np.ix_(ia)] = abs(np.random.normal(new[np.ix_(ia)], variance))
    
    # adjust new as needed and put back into x
    x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending, bounds)
    
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
  assert np.allclose(test1, xinitial.new, atol=1e-04)  
  
  print("\nBest initial score is: {:,}".format(scores[0]))
  

  
  # ================================================================ 
  # Do generations
  # ================================================================ 
  
  Result = []
  for gen in range(maxGenerations):
    
    print (" ------------ gen = {:3d} --------- fit= {:20,.0f} ---".format(gen, Pop[0].Fitness))
    
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
      i0, i1 = np.random.permutation(range(1, popSize))[0:2] # top 50
      if gen%5 == 0:
        v0 = Pop[i0].new.copy()
      else:
        v0 = Pop[0].new.copy()
      v1 = Pop[i1].new.copy()
      
      # make random vector using: new = (v1 -v0) * random + v0
      variance  = np.random.uniform(1e-8, X.Variance)
      ran = np.random.normal(0,variance, len(v0))
      Size = np.random.randint(1, len(v0)+1, 1).item()
      ia = np.sort(np.random.permutation(len(v0))[0:Size])
      new = v0
      new[np.ix_(ia)] = (v1[np.ix_(ia)]-v0[np.ix_(ia)]) * ran[np.ix_(ia)] + v0[np.ix_(ia)]
      new = np.abs(new)
      
      # do a mutation every now and then
      if np.random.rand() < .2:
        tmp = np.random.uniform(0,1, len(v0))
        Size = np.random.randint(1, len(v0)+1, 1).item()
        ia = np.sort(np.random.permutation(len(v0))[0:Size])
        new[np.ix_(ia)] = tmp[np.ix_(ia)]
   
      # adjust new as needed and put back into x
      x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending, bounds)
      
      test1 = np.array([x.__dict__[f] for f in flexibleList])
      assert np.allclose(test1, x.new, atol=1e-04)
    
      # get fitness
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      x.Fitness = fitnessDic['fitness']['total']
      newPop.append(x)

      test2 = np.array([x.__dict__[f] for f in flexibleList])
      assert np.allclose(test2, x.new, atol=1e-04)
    
    Pop = newPop[0:popSize]
    scores = np.array([x.Fitness for x in Pop])
    idd = np.argsort(scores)
    scores = scores[idd]
    Pop = [Pop[i] for i in idd]
    
    if X.doBFGS:
      # call optimize
      x = copy.deepcopy(Pop[0])
      
      test1 = np.array([x.__dict__[f] for f in flexibleList])
      assert np.allclose(test1, x.new, atol=1e-04)
      
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      
      t0 = time.time()
      res = minimize(gfit, x.new, 
        args=(x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending, bounds), 
        method='L-BFGS-B',
        bounds = bounds, 
        options= {
          'disp': False,
          'ftol' : 1e-10, #* np.finfo(float).eps,  #high is weak, fast
          'maxls':200,
          'maxiter':10
          })
      print("BFGS time: {}".format(np.round(time.time()-t0)/60,2))
      new = res['x']
      x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending, bounds)
      x.Fitness = res['fun']
      
      print('bfgs: {:,} vs {:,}, success: {}, Msg: {}\n'.format(x.Fitness, scores[0], res['success'], res['message']))
      
      fitnessDic, _, _ = fitness.getFit(x, stocksDic, Print=False, Optimize=True)
      
      if x.Fitness < scores[0]:
        Pop[0] = x
        scores[0] = x.Fitness 
    
    bestf = scores[0]
    bestx = Pop[0]
    
    Result.append(bestx)
    
    #print("\nGen= ", gen, "score= {:,}".format(scores[0]))
    
    
    if (gen%2 == 0) and (gen > 0):
      flexVars = np.array([x.__dict__[f] for f in flexibleList])
      print("\nFlexible:")
      for ii, f in enumerate(flexibleList):
        print("'{:}': [{:}],".format(f, flexVars[ii]*100))
      print("\n")
      
    
    
    if bestf <= np.minimum(10000, X.population):
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
      args=(x, stocksDic, flexibleList, earmarks, nonmemberSpending, memberSpending, bounds), 
      method='L-BFGS-B',
      bounds = bounds, 
      options= {
          'disp': True,
          'ftol' : 1e-10,  #* np.finfo(float).eps,  #high is weak, fast
          'maxls':200,
          'maxiter':200

        })

    new = res['x']
    x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending, bounds)
    x.Fitness = res['fun']
    print('final bfgs: {:,} vs {:,}, success: {}, Msg: {}\n'.format(x.Fitness, scores[0], res['success'], res['message']))
    if  x.Fitness < scores[0]:
      scores[0] = x.Fitness
      Pop[0] = x  
            
  new = Pop[0].new
  x = copy.deepcopy(Pop[0])
  #x = adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending, bounds)
  
  # run fitness for the winning gene, printing results  
  fitnessDic, tableDic, summaryGraphDic = fitness.getFit(x, stocksDic, Print=True, Optimize=False)

  print('final genetic: ', x.Fitness, " fitdic: ", fitnessDic['fitness']['total'], "\n")
  flexVars = np.array([x.__dict__[f] for f in flexibleList])
  
  flexDic = {}
    
  print("\nFlexible:")
  for ii, f in enumerate(flexibleList):
    flexDic[f] = [flexVars[ii]*100]
    print("['{:}', {:}],".format(f, flexVars[ii]*100))

  print("\nFlexible:")
  for ii, f in enumerate(flexibleList):
    print("'{:}': [{:}],".format(f, flexVars[ii]*100))
  
  fitnessDic['flexDic'] = flexDic
  
  return fitnessDic, tableDic, summaryGraphDic    




#######################################################################################
# helper functions
#######################################################################################


def adjust_flexible_values(x, new, earmarks, nonmemberSpending, memberSpending, bounds):
  """
  Adjusts any values for flexible so that constraints are met
  """
  
  flexibleList = x.flexibleList
  if not isinstance(new, np.ndarray):
    new = np.abs(np.array(new))
  
  # put unadjusted new into x
  _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]

  
  # all must sum to < .999 and be in bounds
  namesDic = {'earmarks': earmarks}
  for groupname in namesDic.keys():  
    names = namesDic[groupname]
    idx = []
    fixed = 0
    for name in names:
      if name in flexibleList:
        idx.append(flexibleList.index(name))
      else:
        fixed += x.__dict__[name]
    
    n = 0
    while True:
      n += 1
      
      unfixed = new[idx].sum()

      if fixed + unfixed > .999:
        mult = (.999 - fixed) / unfixed
        new[idx] = new[idx] * mult  

      # must be in bounds
      for ii in idx:
        new[idx] = np.maximum(bounds[ii][0], new[idx])      
        new[idx] = np.minimum(bounds[ii][1], new[idx])  
      
      if np.any(np.isnan(new)):
        print(groupname, n, idx, new[idx], fixed, unfixed, names) 
        raise Exception

      unfixed = new[idx].sum()
    
      if fixed + unfixed <= .999:
        break
      else:
        #print(groupname, n, idx, new[idx], fixed, unfixed, names)    
        if n > 20:
          print(groupname, n, idx, new[idx], fixed, unfixed, names) 
          raise Exception

  # all pct must sum to 1 and be in bounds
  namesDic = {'memberSpending': memberSpending, 'nonmemberSpending': nonmemberSpending}
  for groupname in namesDic.keys():  
    names = namesDic[groupname]
    if groupname in ['memberSpending', 'nonmemberSpending']:
      # add _pct to name
      names = [i+"_pct" for i in names]
    idx = []
    fixed = 0
    for name in names:
      if name in flexibleList:
        idx.append(flexibleList.index(name))
      else:
        fixed += x.__dict__[name]
    
    n = 0
    while True:
      n += 1
      
      new[idx] = new[idx]
      unfixed = new[idx].sum() 

      if np.allclose(fixed + unfixed, 1, atol=1e-06) == False:
        mult = (1 - fixed) / unfixed
        new[idx] = new[idx] * mult  

      # must be in bounds
      for ii in idx:
        new[idx] = np.maximum(bounds[ii][0], new[idx])      
        new[idx] = np.minimum(bounds[ii][1], new[idx])  
      
      if np.any(np.isnan(new)):
        print(groupname, n, idx, new[idx], fixed, unfixed, names) 
        raise Exception

      unfixed = new[idx].sum()
    
      if np.allclose(fixed + unfixed, 1, atol=1e-06):
        break
      else:
        #print(groupname, n, idx, new[idx], fixed, unfixed, names)    
        if n > 20:
          print("Error: ", groupname, n, idx, new[idx], fixed, unfixed, names) 
          #raise Exception
          break



  # each TS must be in bounds
  namesDic = {'earmarks': earmarks, 'memberSpending': memberSpending}

  for groupname in namesDic.keys():  
    names = namesDic[groupname]
    # add _TS to name
    names = [i+"_TS" for i in names]
    idx = []
    for name in names:
      if name in flexibleList:
        idx.append(flexibleList.index(name))

    # must be in bounds
    for ii in idx:
      new[idx] = np.maximum(bounds[ii][0], new[idx])      
      new[idx] = np.minimum(bounds[ii][1], new[idx])  

    if np.any(np.isnan(new)):
      print(groupname, n, idx, new[idx])
      raise Exception    



  # each earmark_TS must be TSI < ts < .999
  #  idx = []
  #  for name in earmarks:
  #    if name + "_TS" in flexibleList:
  #      idx.append(flexibleList.index(name + "_TS"))
  #  new[idx] = np.minimum(new[idx], .999 )  
  #  new[idx] = np.maximum(new[idx], x.TSI)  





  # put adjusted new into x
  _ = [x.__dict__.__setitem__(flexibleList[i], new[i]) for i in range(new.size)]
  x.new = new
  
  return x


  
