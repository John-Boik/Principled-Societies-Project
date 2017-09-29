
import numpy as np
import networkx as nx
import os, io
from functools import reduce

from leddaApp import app

# ==============================================================================================
np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
np.set_printoptions(6, 120, 150, 220, True)



# ==============================================================================================
# Main function
# ==============================================================================================
def getFit(X, stocksDic, Print=False, Optimize=False):
  """
  Read values from X and incomes and counts from stocksDic to populate token and dollar values
  in save graph. Then calculate the fitness, which is the absolute sum of net flows at all node 
  (should be zero).
  
  Note that G.node[src][dst][0] corresponds to dollar edges and ...[1] corresponds to token edges. 
  """
  
  # open saved networkX graph
  dataFolder = app.config['DATA_FOLDER']
  G = nx.read_graphml(os.path.join(dataFolder, 'steady_state_detailed.graphml'))

  stocksDic = stocksDic['final']  # expected final incomes
    
  # make dict to identify the 9 types of persons, depending on income source
  personStatusDic = {}
  for i, p in enumerate(['person_nonmember_NIWF', 'person_member_NIWF', 
    'person_nonmember_unemployed', 'person_member_unemployed', 
    'person_nonmember_NP', 'person_nonmember_SB',
    'person_member_NP', 'person_member_SB',
    'person_member_PB']):
    personStatusDic[p] = i

  # save in graph the counts and total income per person for each type of person
  personNodes = set(n for n in G.nodes_iter() if G.node[n]['kind']=='person')
  personNodes = np.array(list(personNodes))
  personNodes.sort()   
  
  #print("\nStarting counts and incomes:")
  # note that for nonmember, postCBFS income == income
  wage_target = stocksDic[1]['meanLedda']   
  for p in personNodes:
    if G.node[p]['member'] == 0:
      G.node[p]['count'] = stocksDic[personStatusDic[p]]['cntNonLedda'] 
      G.node[p]['postCBFS_income_mean'] = stocksDic[personStatusDic[p]]['meanNonLedda']
      G.node[p]['postCBFS_income_total'] = stocksDic[personStatusDic[p]]['sumNonLedda']
    else:
      G.node[p]['count'] = stocksDic[personStatusDic[p]]['cntLedda']
      G.node[p]['postCBFS_income_mean'] = stocksDic[personStatusDic[p]]['meanLedda']
      assert wage_target == G.node[p]['postCBFS_income_mean']
      G.node[p]['postCBFS_income_total'] = stocksDic[personStatusDic[p]]['sumLedda']  
    #print("\n", personStatusDic[p], p, "  ", G.node[p])
  #print("\n")
  

  # Nodes  
  nodes = set(n for n in G.nodes_iter())
  nodes = np.array(list(nodes))
  nodes.sort()

  
  # =====================================================================
  #  Edges for person income
  # =====================================================================
  
  # check sum of earmarks
  earmarkSum = X.earmark_SB_subsidy + X.earmark_PB_subsidy + X.earmark_NP_donation + X.earmark_nurture
  assert earmarkSum <= 1
  
  
  # ---------- NIWF, nonmember [0]
  # all support paid by gov
  dollars= G.node['person_nonmember_NIWF']['postCBFS_income_total']
  G.edge['gov']['person_nonmember_NIWF'][0]['value'] = np.maximum(0, dollars)
  
  
  # ---------- NIWF, member [1]
  # gvt support for NIWF LEDDA pays portion of dollar requirement (same as nonmember NIWF), 
  # nurture pays the rest. Adjustment on nurture payments is used to make sure TSI is met.
  cnt = G.node['person_member_NIWF']['count']
  dollars = G.node['person_nonmember_NIWF']['postCBFS_income_mean'] * cnt
  G.edge['gov']['person_member_NIWF'][0]['value'] = np.maximum(0, dollars)
  
  # CBFS support for member NIWF
  dollars = (wage_target * cnt  * (1.-X.TSI) ) / (1 - earmarkSum) - dollars
  tokens = wage_target * X.TSI * cnt / (1 - earmarkSum)
  G.edge['CBFS_nurture']['person_member_NIWF'][0]['value'] = np.maximum(0, dollars)
  G.edge['CBFS_nurture']['person_member_NIWF'][1]['value'] = np.maximum(0, tokens)
  

  # ---------- Unemployed, nonmember [2]
  dollars = G.node['person_nonmember_unemployed']['postCBFS_income_total']
  G.edge['gov']['person_nonmember_unemployed'][0]['value'] = np.maximum(0, dollars)
  
  
  # ---------- unemployed, member [3]                                                        
  # gvt support for unem LEDDA pays portion of dollar requirement, nurture the rest
  cnt = G.node['person_member_unemployed']['count']
  dollars = G.node['person_nonmember_unemployed']['postCBFS_income_mean'] * cnt
  G.edge['gov']['person_member_unemployed'][0]['value'] = np.maximum(0, dollars)
  
  # CBFS, unemployed member
  dollars = (wage_target * cnt  * (1.-X.TSI) ) / (1 - earmarkSum)  - dollars
  tokens = wage_target * X.TSI * cnt / (1 - earmarkSum)
  G.edge['CBFS_nurture']['person_member_unemployed'][0]['value'] = np.maximum(0, dollars)
  G.edge['CBFS_nurture']['person_member_unemployed'][1]['value'] = np.maximum(0, tokens) 
  

  # ----------nonmember NP, nonmember person [4]
  dollars = G.node['person_nonmember_NP']['postCBFS_income_total']
  G.edge['org_nonmember_NP']['person_nonmember_NP'][0]['value'] = np.maximum(0, dollars)
  
  
  # ---------- nonmember SB, nonmember person [5]
  dollars = G.node['person_nonmember_SB']['postCBFS_income_total']
  G.edge['org_nonmember_SB']['person_nonmember_SB'][0]['value'] = np.maximum(0, dollars)


  # ----------member NP, member person [6]
  cnt = G.node['person_member_NP']['count']
  dollars = wage_target * (1.-X.TSI) * cnt / (1 - earmarkSum)
  tokens = wage_target * X.TSI * cnt / (1 - earmarkSum)
  G.edge['org_member_NP']['person_member_NP'][0]['value'] = np.maximum(0, dollars)
  G.edge['org_member_NP']['person_member_NP'][1]['value'] = np.maximum(0, tokens)  

                                                        
  # ----------member SB, member person [7]
  cnt = G.node['person_member_SB']['count']
  dollars = wage_target * (1.-X.TSI) * cnt / (1 - earmarkSum)
  tokens = wage_target * X.TSI * cnt / (1 - earmarkSum)
  G.edge['org_member_SB']['person_member_SB'][0]['value'] = np.maximum(0, dollars)
  G.edge['org_member_SB']['person_member_SB'][1]['value'] = np.maximum(0, tokens)  
  

  # ---------- member PB, member person [8]
  cnt = G.node['person_member_PB']['count']
  dollars = wage_target * (1.-X.TSI) * cnt / (1 - earmarkSum)
  tokens = wage_target * X.TSI * cnt / (1 - earmarkSum)
  G.edge['org_member_PB']['person_member_PB'][0]['value'] = np.maximum(0, dollars)
  G.edge['org_member_PB']['person_member_PB'][1]['value'] = np.maximum(0, tokens)  
  

  # verify TSI
  for node in nodes:
    if (G.node[node]['kind'] == 'person') and (G.node[node]['member']):
      dollars_, tokens_, total_ = sumEdges(G, node, ['in'])  
      if np.isnan(tokens_):
        print("nan in verify TSI: node={:}, G[node]={:}, TSI={:}\n  d,t,total= {:}, {:}, {:}".format(
          node, G.node[node], X.TSI, dollars_, tokens_, total_))
      if np.allclose( np.round(tokens_/total_, 3), X.TSI) == False:
        raise Exception (
          "actual TSI is not X.TSI: {}, ${}, T{}, T&D{}, actual TSI={}, X.TSI={}".format(
          node, dollars_, tokens_, total_, np.round(tokens_/total_, 3), X.TSI))       
      
    
  
  # =====================================================================
  #  Person edges other than person income
  # =====================================================================

  """
  For non-CBFS dollar donations to NP, assume that split between member and nonmember NP
  donations is the same as ratio between expected member/nonmember NP wages paid.
  """
  
  wage_fractions_NP = np.array([
    G.node['person_nonmember_NP']['postCBFS_income_total'],
    G.node['person_member_NP']['postCBFS_income_total']
    ])
  wage_fractions_NP = wage_fractions_NP/wage_fractions_NP.sum()
    
  # keep track of total person spending to orgs in order to measure spending/CBFS revenue partition
  total_person_spending = 0
   
  # NP donations, CBFS contributions, taxes, and spending
  for pn in personNodes:
    # income sum
    dollars, tokens, total = sumEdges(G, pn, ['in']) 
    
    """
    Non-CBFS donations to NP; Assume every member donates dollars, based on dollar income,
    to both member and nonmember NPs.
    These are the only node pairs for which there are 2--3 edges 
    """
    assert G.edge[pn]['org_nonmember_NP'][1]['kind'] == 'regular_donation'
    G.edge[pn]['org_nonmember_NP'][1]['value'] = np.maximum(0, dollars * wage_fractions_NP[0] * \
      X.Config.NP_reg_donation_pct_dollar_income)    
    if G.node[pn]['member']:
      # spends tokens and dollars, plus donation dollars for three edges
      assert G.edge[pn]['org_member_NP'][2]['kind'] == 'regular_donation'    
      G.edge[pn]['org_member_NP'][2]['value'] = np.maximum(0, dollars * wage_fractions_NP[1] * \
        X.Config.NP_reg_donation_pct_dollar_income)
    else:
      assert G.edge[pn]['org_member_NP'][1]['kind'] == 'regular_donation' 
      G.edge[pn]['org_member_NP'][1]['value'] = np.maximum(0, dollars * wage_fractions_NP[1] * \
        X.Config.NP_reg_donation_pct_dollar_income)
          
    # CBFS contributions
    if G.node[pn]['member']:
      
      # SB CBFS contributions-----------------------------------
      dollars = total * X.earmark_SB_subsidy * (1 - X.earmark_SB_subsidy_TS)
      tokens  = total * X.earmark_SB_subsidy * (X.earmark_SB_subsidy_TS)
      G.edge[pn]['CBFS_SB_subsidy'][0]['value'] = np.maximum(0, dollars)
      G.edge[pn]['CBFS_SB_subsidy'][1]['value'] = np.maximum(0, tokens)
      
      # PB CBFS contributions-----------------------------------
      dollars = total * X.earmark_PB_subsidy * (1 - X.earmark_PB_subsidy_TS)
      tokens  = total * X.earmark_PB_subsidy * (X.earmark_PB_subsidy_TS)
      G.edge[pn]['CBFS_PB_subsidy'][0]['value'] = np.maximum(0, dollars)
      G.edge[pn]['CBFS_PB_subsidy'][1]['value'] = np.maximum(0, tokens)      

      # NP CBFS contributions-----------------------------------
      dollars = total * X.earmark_NP_donation * (1 - X.earmark_NP_donation_TS)
      tokens  = total * X.earmark_NP_donation * (X.earmark_NP_donation_TS)
      G.edge[pn]['CBFS_NP_donation'][0]['value'] = np.maximum(0, dollars)
      G.edge[pn]['CBFS_NP_donation'][1]['value'] = np.maximum(0, tokens)   

      # Nurture CBFS contributions-----------------------------------
      dollars = total * X.earmark_nurture * (1 - X.earmark_nurture_TS)
      tokens  = total * X.earmark_nurture * (X.earmark_nurture_TS)
      G.edge[pn]['CBFS_nurture'][0]['value'] = np.maximum(0, dollars)
      G.edge[pn]['CBFS_nurture'][1]['value'] = np.maximum(0, tokens)   


    # pay taxes based on total income, minus deductions
    tax = getTax(G, pn, total, X.Config.gov_tax_rate, X.Config.gov_tax_standard_deduction)  
    G.edge[pn]['gov'][0]['value'] = tax
     

    # personal spending: any remaining currency in person node is spent to orgs
    dollars, tokens, total = sumEdges(G, pn, ['both']) 
    
    dollars = np.maximum(0, dollars)
    tokens  = np.maximum(0, tokens)
    total   = dollars + tokens
    
    
    for org in ['org_member_NP', 'org_member_PB', 'org_member_SB', 'org_nonmember_NP', 
      'org_nonmember_SB']:
        
      if not G.node[pn]['member']:
        assert np.allclose(dollars, total)
        spendkey = 'person_member_spending_to_' + org[4:] + "_pct"
        dollars_ = total * X.__dict__[spendkey]
        total_person_spending += dollars_
        G.edge[pn][org][0]['value'] = np.maximum(0, dollars_)
      
      else:
        # is member
        if 'nonmember' in org:
          # only dollars to nonmember orgs
          spendkey = 'person_member_spending_to_' + org[4:] + "_pct"
          dollars_ = total * X.__dict__[spendkey] 
          total_person_spending += dollars_
          G.edge[pn][org][0]['value'] =  np.maximum(0, dollars_)
        else:
          # member person and member org: tokens and dollars
          spendkey = 'person_member_spending_to_' + org[4:] + "_pct"
          sharekey = 'person_member_spending_to_' + org[4:] + "_TS"
          total_ = total * X.__dict__[spendkey] 
          tokens_  = total_ * X.__dict__[sharekey]      
          dollars_ = total_ * (1- X.__dict__[sharekey])
          total_person_spending += total_
          G.edge[pn][org][0]['value'] =  np.maximum(0, dollars_)
          G.edge[pn][org][1]['value'] =  np.maximum(0, tokens_)         

                                            
  # =====================================================================
  #  Gov edges to organizations
  # =====================================================================  
  
  """
  Gov subsidies and contracts can go to any NP, SB, or PB.  There are 2 kinds:
  gov_grants_contracts (for NP) and gov_subsidies_contracts (for SB, PB). 
  For each of these, assume that the gov support is split up according to
  the ratio between the different groups wages paid.  Here, FP= for profit (SB+PB).
  """

  wage_fractions_FP = np.array([
    G.node['person_nonmember_SB']['postCBFS_income_total'],
    G.node['person_member_SB']['postCBFS_income_total'],
    G.node['person_member_PB']['postCBFS_income_total']
    ])
  wage_fractions_FP = wage_fractions_FP/wage_fractions_FP.sum()

  NP_gov_spending = X.Gov.grant_NP_dollars_annual + X.Gov.contract_NP_dollars_annual
  FP_gov_spending = X.Gov.contract_forprofit_dollars_annual + X.Gov.subsidy_forprofit_dollars_annual
  
  
  G.edge['gov']['org_nonmember_NP'][0]['value'] = NP_gov_spending * wage_fractions_NP[0]
  G.edge['gov']['org_member_NP'][0]['value']    = NP_gov_spending * wage_fractions_NP[1]
  
  G.edge['gov']['org_nonmember_SB'][0]['value'] = FP_gov_spending * wage_fractions_FP[0]
  G.edge['gov']['org_member_SB'][0]['value']    = FP_gov_spending * wage_fractions_FP[1]
  G.edge['gov']['org_member_PB'][0]['value']    = FP_gov_spending * wage_fractions_FP[2]


  # gov to Roc. Gov will spend its balance on roc
  dollars, tokens, total = sumEdges(G, 'gov', ['both']) 
  assert tokens == 0
  dollars = np.maximum(0, dollars)
  G.edge['gov']['roc'][0]['value'] = dollars

  
  # =====================================================================
  #  CBFS to organizations
  # ===================================================================== 
  
  # keep track of total CBFS spending to orgs in order to measure spending/CBFS revenue partition
  total_CBFS_spending = 0
    
  dollars, tokens, total = sumEdges(G, 'CBFS_SB_subsidy', ['in']) 
  dollars = np.maximum(0, dollars)
  tokens  = np.maximum(0, tokens)
  total   = dollars + tokens
  total_CBFS_spending += total
  
  G.edge['CBFS_SB_subsidy']['org_member_SB'][0]['value'] = dollars
  G.edge['CBFS_SB_subsidy']['org_member_SB'][1]['value'] = tokens
  
  dollars, tokens, total = sumEdges(G, 'CBFS_PB_subsidy', ['in']) 
  dollars = np.maximum(0, dollars)
  tokens  = np.maximum(0, tokens)
  total   = dollars + tokens
  total_CBFS_spending += total

  G.edge['CBFS_PB_subsidy']['org_member_PB'][0]['value'] = dollars
  G.edge['CBFS_PB_subsidy']['org_member_PB'][1]['value'] = tokens  

  dollars, tokens, total = sumEdges(G, 'CBFS_NP_donation', ['in']) 
  dollars = np.maximum(0, dollars)
  tokens  = np.maximum(0, tokens)
  total   = dollars + tokens
  total_CBFS_spending += total

  G.edge['CBFS_NP_donation']['org_member_NP'][0]['value'] = dollars
  G.edge['CBFS_NP_donation']['org_member_NP'][1]['value'] = tokens  

  # add to fitness total the difference for revenue partition
  try:
    ratio = X.CBFS_spending_ratio
    expected_CBFS = ratio * (total_person_spending + total_CBFS_spending)
    delta_CBFS_ratio = int(round(abs(total_CBFS_spending - expected_CBFS),0)) 
    #print("difference for CBFS/person spending ratio = {:.2E}".format(delta_CBFS_ratio))
  except:
    delta_CBFS_ratio = 0 


  # =====================================================================
  #  trade balance, org <--> roc
  # ===================================================================== 
  
  """
  Gov spent a lump sum of dollars into RoC. Now each Org has to recover a portion
  so that in total, all is recovered. In the ideal, this completes all flows
  and all flows at each node sum to 0.  To do this simply, order Orgs in list and 
  allot in sequential order the amount each needs to reach a balance of zero. Stop
  when RoC funds run dry.
  
  Note that purchases from RoC is not modeled here, because purchases would offset sales.
  Rather, the net revenue gain from RoC is modeled (for accounting purposes, purchases are zero).
  
  """

  dollars, tokens, total = sumEdges(G, 'roc', ['in']) 
  dollar_pool = np.maximum(0, dollars)
  assert tokens == 0
  
  for org in ['org_member_NP', 'org_member_PB', 'org_member_SB', 'org_nonmember_NP', 
    'org_nonmember_SB']:

    dollars_, tokens_, total_ = sumEdges(G, org, ['both'])  
    if dollars_ < 0:
      available = np.minimum(np.abs(dollars_), dollar_pool)
      G.edge['roc'][org][0]['value'] = available
      dollar_pool -= available
      assert dollar_pool >= 0
  
  # calculate fitness and collect node and edge values in dict
  fitnessDic = calc_fitness(G, nodes, Optimize, Print, delta_CBFS_ratio)
  
  if Optimize:
    summaryGraphDic = None
    tableDic = None
  else:
    # summarize the fitness data for the summary graph
    summaryGraphDic = summaryGraph(fitnessDic, G)
  
    # generate data for node tables
    tableDic = save_node_tables(G, fitnessDic, nodes)

  if 1==2:
    # for testing purposes  
    print("\n=================\nFitness Dict:")
    print("\n\nFitness: ", fitnessDic['fitness'])
    
    print("\n\nClusters: ")
    keys = list(fitnessDic['clusters'].keys())
    keys.sort()
    for k in keys:
      print("  ", k, ": ", fitnessDic['clusters'][k])
      
    print("\n\nNodes: ")
    keys = list(fitnessDic['nodes'].keys())
    keys.sort()
    for k in keys:
      print("  ", k, ": ", fitnessDic['nodes'][k])  

    print("\n\nEdges: ")
    keys = list(fitnessDic['edges'].keys())
    keys.sort()
    for k in keys:
      print("  ", k, ": ", fitnessDic['edges'][k])  
      


  if not Optimize:
    # create a flexDic to hold returned, unchanged parameters
    flexibleList = X.flexibleList
    flexVars = np.array([X.__dict__[f] for f in flexibleList])
    flexDic = {}
        
    for ii, f in enumerate(flexibleList):
      flexDic[f] = [flexVars[ii]*100]
      #print("'{:}': [{:}],".format(f, flexVars[ii]*100))
    
    fitnessDic['flexDic'] = flexDic
    
 
  return fitnessDic, tableDic, summaryGraphDic

  

# ==============================================================================================
# Helper functions
# ==============================================================================================

def getSums(G, edgeList, Print=False):
  """
  Sum inflows or outflows around a list of nodes for tokens, dollars, and total T&D. 
  """
  dollars = 0
  tokens = 0
  
  for src, dst in edgeList:
    for i in G.edge[src][dst]:
      if G.edge[src][dst][i]['unit'] == 'dollars':
        dollars += G.edge[src][dst][i]['value']
      else:
        assert G.edge[src][dst][i]['unit'] == 'tokens'
        tokens += G.edge[src][dst][i]['value']
  
  total = dollars + tokens
  return (dollars, tokens, total)


# =====================================================================
def sumEdges(G, nodes, directions, Print=False):
  """
  Sum inflows and/or outflows around a node for tokens, dollars, and total T&D
  directions is a list of ['in', 'out', 'both'], or for both ['in', 'out'].
  nodes can be a single node or a list [node1, node2, etc.].
  """
  tokens = 0
  dollars = 0
  total = 0
  
  if not isinstance(directions, list):
    directions = [directions]
  if not isinstance(nodes, list):
    nodes = [nodes]

  if ('in' in directions) or ('both' in directions):  
    edgeList = []
    for node in nodes:
      edgeList.extend(list(set(G.in_edges([node]))))
      
    dollars_, tokens_, total_ = getSums(G, edgeList, Print=False)
    dollars += dollars_
    tokens += tokens_
    total += total_

  if ('out' in directions) or ('both' in directions):  
    edgeList = []
    for node in nodes:
      edgeList.extend(list(set(G.out_edges([node]))))
    
    dollars_, tokens_, total_ = getSums(G, edgeList, Print=False)
    dollars -= dollars_
    tokens -= tokens_
    total -= total_
        
  return (dollars, tokens, total)


# ===================================================================== 
def getTax(G, node, total, taxRate, deduction):
  """
  Calculate taxes due
  """
  agi = total - (deduction * G.node[node]['count'])
  
  # deduct donation to NP, and CBFS contributions for NP and nurture
  agi -= G.edge[node]['org_nonmember_NP'][0]['value']
  agi -= G.edge[node]['org_member_NP'][0]['value']
  
  if G.node[node]['member']:
    agi -= G.edge[node]['CBFS_NP_donation'][0]['value']
    agi -= G.edge[node]['CBFS_NP_donation'][1]['value']
    agi -= G.edge[node]['CBFS_nurture'][0]['value']
    agi -= G.edge[node]['CBFS_nurture'][1]['value']    
    
  agi = np.maximum(0, agi)    
  tax = agi * taxRate  
  
  return tax


# ===================================================================== 
def calc_fitness(G, nodes, Optimize, Print, delta_CBFS_ratio):
  """
  Calculates fitness and collects node and edge values in dict
  """
  
  fitnessDic = {}

  # edges
  edgeDic = {}
  for src in nodes:
    for dst in nodes:
      if src==dst:
        # no self loops
        continue
      try:
        edge = G.edge[src][dst]
      except:
        continue
      for i in edge:
        ID = edge[i]['id']
        edgeDic[ID] = edge[i] 
        edgeDic[ID].update({'src':src, 'dst':dst})
        edgeDic[ID].update({'value': int(round(edgeDic[ID]['value']))})
  
  # nodes
  fitness_dollars = 0
  fitness_tokens  = 0
  nodeDic = {}
  for node in nodes:
    dollars, tokens, total = sumEdges(G, node, ['both']) 
    dollars = int(round(np.abs(dollars)))
    tokens = int(round(np.abs(tokens)))
    total = dollars + tokens
    
    fitness_dollars += dollars
    fitness_tokens += tokens
        
    ID = G.node[node]['id']
    nodeDic[ID] = G.node[node] 
    nodeDic[ID].update({'fitness_dollars':dollars, 'fitness_tokens':tokens, 
      'fitness_total':total, 'name':node})
    
    # only do if not Optimize
    if not Optimize and (G.node[node]['kind'] == 'person'):
      income_dollars, income_tokens, income_total = sumEdges(G, node, ['in']) 
      
      # calculate pre- and post-CBFS income
      CBFS_dollars = 0
      CBFS_tokens = 0
      
      if G.node[node]['member']:
        # subtract CBFS contributions
        outEdges = list(set(G.out_edges([node])))
        for e0, e1 in outEdges:
          for i in G.edge[e0][e1]:
            if G.edge[e0][e1][i]['kind'] in ["contribution"]:
              if G.edge[e0][e1][i]['unit'] == 'dollars':
                CBFS_dollars += G.edge[e0][e1][i]['value']
              else:
                assert G.edge[e0][e1][i]['unit'] == 'tokens'
                CBFS_tokens += G.edge[e0][e1][i]['value']    
          
      cnt = G.node[node]['count']
      nodeDic[ID].update({
        'actual_preCBFS_income_dollars': income_dollars, 
        'actual_preCBFS_income_tokens': income_tokens,
        'actual_preCBFS_income_total': income_dollars + income_tokens,
        
        'actual_postCBFS_income_dollars': income_dollars - CBFS_dollars, 
        'actual_postCBFS_income_tokens': income_tokens - CBFS_tokens,
        'actual_postCBFS_income_total': income_dollars + income_tokens - CBFS_dollars - CBFS_tokens,
        
        'actual_preCBFS_income_dollars_mean': int(round((income_dollars)/cnt)), 
        'actual_preCBFS_income_tokens_mean': int(round((income_tokens)/cnt)),
        'actual_preCBFS_income_total_mean': int(round((income_dollars + income_tokens)/cnt)),
        
        'actual_postCBFS_income_dollars_mean': int(round((income_dollars - CBFS_dollars)/cnt)), 
        'actual_postCBFS_income_tokens_mean': int(round((income_tokens - CBFS_tokens)/cnt)),
        'actual_postCBFS_income_total_mean': int(round(
          (income_dollars + income_tokens - CBFS_dollars - CBFS_tokens)/cnt))
        })
      

      actual_postCBFS = int(round(nodeDic[ID]['actual_postCBFS_income_total']))
      expected_postCBFS = int(round(G.node[node]['postCBFS_income_total']))
      actual_postCBFS_mean = int(round(nodeDic[ID]['actual_postCBFS_income_total_mean']))
      expected_postCBFS_mean = int(round(G.node[node]['postCBFS_income_mean']))
      
      if (np.allclose(expected_postCBFS, actual_postCBFS) == False):
        print(("""
          **** Post-CBFS income unmatched for node '{:s}': 
            actual= {:,.0f},
            expected= {:,.0f}, 
            delta= {:,.0f}""").format(node, actual_postCBFS, 
            expected_postCBFS, actual_postCBFS - expected_postCBFS))
        raise Exception() 


      if (Print) and (np.allclose(expected_postCBFS_mean, actual_postCBFS_mean) == False):
        print(("""
          **** Post-CBFS mean income unmatched for node '{:s}': 
            actual= {:,.0f},
            expected= {:,.0f}, 
            delta= {:,.0f}""").format(node, actual_postCBFS_mean, 
            expected_postCBFS_mean, actual_postCBFS_mean - expected_postCBFS_mean))      
        raise Exception() 
  
  fitnessDic['fitness'] = {'dollars': int(round(fitness_dollars)), 
    'tokens': int(round(fitness_tokens)), 
    'total': int(round(fitness_dollars + fitness_tokens + delta_CBFS_ratio))}
  
  if Optimize:
    return fitnessDic 
  
  # clusters (code copied from "make_detailed_steady_state_graph.py" results)
  clusterDic = {}

  clusterDic['c10'] = {'name':'cluster_CBFS', 
    'nodes':['CBFS_PB_subsidy', 'CBFS_SB_subsidy', 'CBFS_NP_donation', 'CBFS_nurture']}

  clusterDic['c07'] = {'name':'cluster_persons_unemployed', 'nodes':[]}
  clusterDic['c09'] = {'name':'cluster_persons_unemployed_nonmember', 
    'nodes':['person_nonmember_NIWF', 'person_nonmember_unemployed']}
  clusterDic['c08'] = {'name':'cluster_persons_unemployed_member', 
    'nodes':['person_member_unemployed', 'person_member_NIWF']}

  clusterDic['c04'] = {'name':'cluster_persons_employed', 'nodes':[]}
  clusterDic['c05'] = {'name':'cluster_persons_employed_member', 
    'nodes':['person_member_NP', 'person_member_SB', 'person_member_PB']}
  clusterDic['c06'] = {'name':'cluster_persons_employed_nonmember', 
    'nodes':['person_nonmember_SB', 'person_nonmember_NP']}

  clusterDic['c01'] = {'name':'cluster_orgs', 'nodes':[]}
  clusterDic['c03'] = {'name':'cluster_orgs_nonmember', 
    'nodes':['org_nonmember_SB', 'org_nonmember_NP']}
  clusterDic['c02'] = {'name':'cluster_orgs_member', 
    'nodes':['org_member_PB', 'org_member_SB', 'org_member_NP']}


  # adjustment nodes for empty primary clusters
  clusterDic['c07']['nodes'] = clusterDic['c09']['nodes'] + clusterDic['c08']['nodes']
  clusterDic['c04']['nodes'] = clusterDic['c05']['nodes'] + clusterDic['c06']['nodes']
  clusterDic['c01']['nodes'] = clusterDic['c03']['nodes'] + clusterDic['c02']['nodes']

  for k in clusterDic.keys():
    dollars, tokens, total = sumEdges(G, clusterDic[k]['nodes'], ['both'])
    clusterDic[k].update({'fitness_dollars':int(round(dollars)), 'fitness_tokens':int(round(tokens)),
      'fitness_total':int(round(total))}) 
    
  fitnessDic['clusters'] = clusterDic
  fitnessDic['nodes'] = nodeDic
  fitnessDic['edges'] = edgeDic
  
  return fitnessDic        
        


# ===================================================================== 
def summaryGraph(fitnessDic, G):
  """
  Return a dict that collects fitness data for use with the summary svg graph
  
  This function is unfinished.
  """  
  
  edges = {}
  nodes = {}
  
  edges['path_Org_wages'] = 0
  edges['path_Persons_Emp_spending'] = 0
  edges['path_Persons_Unemp_spending'] = 0
  edges['path_Persons_Emp_contributions'] = 0
  edges['path_Persons_Unemp_contributions'] = 0 
  edges['path_Nurture_support'] = 0
  edges['path_Gov_support'] = 0
  edges['path_Persons_Emp_taxes'] = 0
  edges['path_Persons_Unemp_taxes'] = 0 
  edges['path_Gov_Orgs'] = 0
  edges['path_Gov_ROC'] = 0
  edges['path_Org_sales'] = 0
  # edges['path_Org_purchases'] = 0   # assumed to be zero, because net trade is modeled
  edges['path_CBFS_Orgs'] = 0
  edges['path_Persons_Emp_Donations'] = 0
  edges['path_Persons_Unemp_Donations'] = 0
  
     
  edgeDic = fitnessDic['edges']
  keys = list(edgeDic.keys())
  for e in keys:
    ed = edgeDic[e]
    src = ed['src']
    dst = ed['dst']
    kind = ed['kind']
 
    if kind == 'wage':
      edges['path_Org_wages'] += ed['value']  

    elif kind == 'spending':
      if G.node[src]['employed']:
        edges['path_Persons_Emp_spending'] += ed['value']
      else:
        edges['path_Persons_Unemp_spending'] += ed['value']  
            
    elif kind == 'contribution':
      if G.node[src]['employed']:
        edges['path_Persons_Emp_contributions'] += ed['value']
      else:
        edges['path_Persons_Unemp_contributions'] += ed['value']      
        
    elif kind == 'nurture':
      edges['path_Nurture_support'] += ed['value']

    elif kind == 'gov_support':
      edges['path_Gov_support'] += ed['value']

    elif kind == 'taxes':
      if G.node[src]['employed']:
        edges['path_Persons_Emp_taxes'] += ed['value']
      else:
        edges['path_Persons_Unemp_taxes'] += ed['value']  

    elif (kind == 'gov_subsidies_contracts') or (kind == 'gov_grants_contracts'):
      edges['path_Gov_Orgs'] += ed['value']
    
    elif kind == 'gov_spending':
      edges['path_Gov_ROC'] += ed['value']
    
    elif kind == 'trade':
      edges['path_Org_sales'] += ed['value']      

    elif kind == 'CBFS_funding':
      edges['path_CBFS_Orgs'] += ed['value']      

    elif kind == 'regular_donation':  
      if G.node[src]['employed']:
        edges['path_Persons_Emp_Donations'] += ed['value']
      else:
        edges['path_Persons_Unemp_Donations'] += ed['value']
    else:
      print (ed)
      raise Exception()
        
  summaryGraphDic = {'edges':edges, 'nodes':nodes}
  return summaryGraphDic




# ===================================================================== 
def save_node_tables(G, fitnessDic, nodes):
  """
  Save a table showing inflows and outflows to each node
  """  
  
  PrintTables = False
  
  Tables = io.StringIO()
  divs = 1  # units, millions, billions, etc.  
  tableDic = {}
  
  TOKENS = 0
  DOLLARS = 0
  TOTAL = 0
  
  for node in nodes:
    ID = G.node[node]['id']
    tableDic[ID] =  { 'in': {'Sums':'', 'Values':''}, 
                      'out':{'Sums':'', 'Values':''}, 
                      'grandSums':'', 
                      'name':node}
    
    # flows ------------------------------------------------------------------
    d = {'in':{}, 'out':{}}
    for flow in ['in', 'out']:
      if flow == 'in':
        Edges = list(set(G.in_edges([node])))
      else:
        Edges = list(set(G.out_edges([node])))
      
      for e0, e1 in Edges:
        if flow == 'in':
          label = e0
        else:
          label = e1       
        
        if label not in d[flow].keys():
          d[flow][label] = {}
        
        for i in G.edge[e0][e1]:
          kind = G.edge[e0][e1][i]['kind']
          if kind not in d[flow][label].keys():
            d[flow][label][kind] = [0,0]
          
          if G.edge[e0][e1][i]['unit'] == 'dollars':
            d[flow][label][kind][0] = G.edge[e0][e1][i]['value'] / divs
          else:
            assert G.edge[e0][e1][i]['unit'] == 'tokens'
            d[flow][label][kind][1] = G.edge[e0][e1][i]['value'] / divs
          
      Values = []
      SumDollars = 0
      SumTokens = 0
      for label in d[flow].keys():
        for kind in d[flow][label].keys():  
          dollars, tokens = d[flow][label][kind]
          total = dollars + tokens
          SumDollars += dollars
          SumTokens += tokens
          Values.append([kind, label, int(dollars), int(tokens), int(total)])
      
      # create 2D array, sort by first two columns
      tmp = np.array(Values)
      idx = np.lexsort((tmp[:,1], tmp[:,0]))
      Values = [Values[i] for i in idx]
      tableDic[ID][flow]['Values'] = Values
      tableDic[ID][flow]['Sums'] = ["Subtotal "+flow+":", " ", int(SumDollars), int(SumTokens), 
        int(SumDollars + SumTokens)]
          
    # now add grandtotals 
    grandDollars = tableDic[ID]['in']['Sums'][2] - tableDic[ID]['out']['Sums'][2]
    grandTokens  = tableDic[ID]['in']['Sums'][3] - tableDic[ID]['out']['Sums'][3]
    grandTotal   = tableDic[ID]['in']['Sums'][4] - tableDic[ID]['out']['Sums'][4]
    tableDic[ID]['grandSums'] = ['Grand Total:', '', int(grandDollars), int(grandTokens),
      int(grandTotal)] 
    
    TOKENS += abs(grandTokens)
    DOLLARS += abs(grandDollars)
    TOTAL += abs(grandTokens) + abs(grandDollars)
    
    # print out tables
    if PrintTables:
      Tables.write("\n====================\nNode= {}\n".format(node))
      Tables.write("\nIn Flows:\n")
      for i in tableDic[ID]['in']['Values']:
        Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n".format(*i))
      Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n\n".format(*tableDic[ID]['in']['Sums'])) 

      Tables.write("Out Flows:\n")
      for i in tableDic[ID]['out']['Values']:
        Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n".format(*i))
      Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n\n".format(*tableDic[ID]['out']['Sums'])) 
      
      Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n\n".format(*tableDic[ID]['grandSums'])) 
      
      if G.node[node]['kind'] == 'person':
        # also print per person in and out
        count = G.node[node]['count'] / divs
        
        Tables.write("\nIn Flows (per person):\n")
        for i in tableDic[ID]['in']['Values']:
          Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n".format(
            *[v if isinstance(v,str) else v/count for v in i]))
        Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n\n".format(
          *[v if isinstance(v,str) else v/count for v in tableDic[ID]['in']['Sums']]))

        Tables.write("Out Flows (per person):\n")
        for i in tableDic[ID]['out']['Values']:
          Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n".format(
            *[v if isinstance(v,str) else v/count for v in i]))
        Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n\n".format(
          *[v if isinstance(v,str) else v/count for v in tableDic[ID]['out']['Sums']]))
        
        Tables.write("{:<24s} {:<28s} {:>16,.0f} {:>16,.0f} {:>16,.0f}\n\n".format(
          *[v if isinstance(v,str) else v/count for v in tableDic[ID]['grandSums']]))

  if PrintTables:
    Tables.write("\n========== Grand Total Fitness ==========\n")
    Tables.write("Dollars: {:,d}  Tokens: {:,d}  Total: {:,d}\n".format(int(DOLLARS*divs),
      int(TOKENS*divs), int(TOTAL*divs)))
    T = Tables.getvalue()
    print(T)  
  
  return tableDic


