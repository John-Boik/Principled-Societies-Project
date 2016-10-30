
"""
This script uses networkx to create and save (as graphml) a base ~20-node graph for the steady 
state model. All currency elements of the graph are set to zero. The script also uses the 
networkx graph to create and save a svg file, using graphviz (via pydot).   

To run:
  at python interactive cursor:
    exec(open('./make_detailed_steady_state_graph.py').read()), or 
  in command window:
    python make_detailed_steady_state_graph.py
  
After creating the graphml and svg files, copy them to the /static/data and /static/images folders,
respectively.
  
"""


# ============================== Imports ==========================================================
import numpy as np
import networkx as nx
import pydot
import shutil

# =============================== Intialization ===================================================

graphml_filename = 'steady_state_detailed.graphml'
svg_filename = 'steady_state_detailed.svg'

G = nx.MultiDiGraph()


##############################################################################################
#  Make graphx nodes
##############################################################################################

counter = (i for i in range(10000))

# person nodes, employment
for member in [True, False]:
  for group in ['NP', 'SB', 'PB', 'NIWF', 'unemployed' ]:
    if (member == False) and (group in ['PB']):
      continue
    
    employed = True
    if group in ['NIWF', 'unemployed']:
      employed = False
     
    name = "person" + ("_member_" if member==True else "_nonmember_") + group  
    G.add_node(name, kind='person', member=member, employed=employed, group=group, id=next(counter))


# org nodes
for member in [True, False]:
  for group in ['NP', 'SB', 'PB']:
    if (member == False) and (group in ['PB']):
      continue   
    
    name = "org" + ("_member_" if member==True else "_nonmember_") + group 
    G.add_node(name, kind='org', member=member, group=group, id=next(counter))


# CBFS nodes
for group in ['SB_subsidy', 'PB_subsidy', 'NP_donation', 'nurture']:
  
  name = "CBFS_" + group
  G.add_node(name, kind='CBFS', group=group, id=next(counter))


# gov node
G.add_node("gov", kind="gov", id=next(counter))

  
# Rest of counties
G.add_node("roc", kind="roc", id=next(counter))


##############################################################################################
#  Make graphx edges
##############################################################################################

# Edges: person <--> org, income and spending
personNodes = set(n for n in G.nodes_iter() if G.node[n]['kind']=='person')
orgNodes =    set(n for n in G.nodes_iter() if G.node[n]['kind']=='org')

for org in orgNodes:
  for person in personNodes:
    
    # income
    if (G.node[person]['group'] == G.node[org]['group']) and \
        (G.node[person]['member'] == G.node[org]['member']):
      G.add_edge(org, person, unit='dollars', kind="wage", value=0, id=next(counter))
      if G.node[person]['member']:
        G.add_edge(org, person, unit='tokens',  kind="wage", value=0, id=next(counter))
    
    # spending
    G.add_edge(person, org, unit='dollars', kind="spending", value=0, id=next(counter))
    if G.node[person]['member'] and G.node[org]['member']:
      G.add_edge(person, org, unit='tokens',  kind="spending", value=0, id=next(counter))    
    
    # donations to NP
    if G.node[org]['group'] == "NP":
      G.add_edge(person, org, unit='dollars', kind="regular_donation", value=0, id=next(counter))      

    
# Edges: gov <--> person
for person in personNodes:
  
  #support
  if (G.node[person]['group'] in ['NIWF', 'unemployed']):
    G.add_edge('gov', person, unit='dollars', kind="gov_support", value=0, id=next(counter))
  
  # taxes
  G.add_edge(person, 'gov', unit='dollars', kind="taxes", value=0, id=next(counter))    


# Edges: CBFS <--> person
for person in personNodes:
  if (G.node[person]['group'] in ['NIWF', 'unemployed']) and G.node[person]['member']:
    G.add_edge('CBFS_nurture', person, unit='dollars', kind="nurture", value=0, id=next(counter))
    G.add_edge('CBFS_nurture', person, unit='tokens', kind="nurture", value=0, id=next(counter))
    
  if G.node[person]['member']:
    for group in ['CBFS_SB_subsidy', 'CBFS_PB_subsidy', 'CBFS_NP_donation', 'CBFS_nurture']:
      G.add_edge(person, group, unit='dollars', kind="contribution", value=0, id=next(counter))
      G.add_edge(person, group, unit='tokens',  kind="contribution", value=0, id=next(counter))

# Edges: CBFS --> orgs
G.add_edge('CBFS_SB_subsidy', 'org_member_SB', unit='dollars', kind="CBFS_funding", value=0, id=next(counter))
G.add_edge('CBFS_SB_subsidy', 'org_member_SB', unit='tokens',  kind="CBFS_funding", value=0, id=next(counter))

G.add_edge('CBFS_PB_subsidy', 'org_member_PB', unit='dollars', kind="CBFS_funding", value=0, id=next(counter))
G.add_edge('CBFS_PB_subsidy', 'org_member_PB', unit='tokens',  kind="CBFS_funding", value=0, id=next(counter))

G.add_edge('CBFS_NP_donation', 'org_member_NP', unit='dollars', kind="CBFS_funding", value=0, id=next(counter))
G.add_edge('CBFS_NP_donation', 'org_member_NP', unit='tokens',  kind="CBFS_funding", value=0, id=next(counter))


# Edges: gov --> orgs
G.add_edge('gov', 'org_member_SB',    unit='dollars', kind="gov_subsidies_contracts", value=0, id=next(counter))
G.add_edge('gov', 'org_member_PB',    unit='dollars', kind="gov_subsidies_contracts", value=0, id=next(counter))
G.add_edge('gov', 'org_member_NP',    unit='dollars', kind="gov_grants_contracts",    value=0, id=next(counter))
G.add_edge('gov', 'org_nonmember_SB', unit='dollars', kind="gov_subsidies_contracts", value=0, id=next(counter))
G.add_edge('gov', 'org_nonmember_NP', unit='dollars', kind="gov_grants_contracts",    value=0, id=next(counter))


# Edges: gov --> roc
G.add_edge('gov', 'roc', unit='dollars', kind="gov_spending", value=0, id=next(counter))


# Edges: orgs <--> roc
#G.add_edge('org_member_SB', 'roc', unit='dollars', kind="trade", value=0, id=next(counter))
#G.add_edge('org_member_PB', 'roc', unit='dollars', kind="trade", value=0, id=next(counter))
#G.add_edge('org_member_NP', 'roc', unit='dollars', kind="trade", value=0, id=next(counter))
#G.add_edge('org_nonmember_SB', 'roc', unit='dollars', kind="trade", value=0, id=next(counter))
#G.add_edge('org_nonmember_NP', 'roc', unit='dollars', kind="trade", value=0, id=next(counter))

G.add_edge('roc', 'org_member_SB', unit='dollars', kind="trade", value=0, id=next(counter))
G.add_edge('roc', 'org_member_PB', unit='dollars', kind="trade", value=0, id=next(counter))
G.add_edge('roc', 'org_member_NP', unit='dollars', kind="trade", value=0, id=next(counter))
G.add_edge('roc', 'org_nonmember_SB', unit='dollars', kind="trade", value=0, id=next(counter))
G.add_edge('roc', 'org_nonmember_NP', unit='dollars', kind="trade", value=0, id=next(counter))


#####################################################################################
#  Print networkx graph elements and save results
#####################################################################################

# print nodes
nodes = set(n for n in G.nodes_iter())
nodes = np.array(list(nodes))
nodes.sort()

print("\nCount of elements (nodes+edges) = ", next(counter))


print("\nNodes:")
for n in nodes:
  print ("node: ", n, "  ", G.node[n])


# print edges
#edges = set(n for n in G.edges_iter())
#edges = np.array(list(edges))

edgeLabelDic = {}
print("\nEdges:")
for src in nodes:
  for dst in nodes:
    if src==dst:
      # no self loops
      continue
    try:
      edge = G.edge[src][dst]
    except:
      continue
    
    print("\nedge: ", src, "---->", dst)
    for i in edge:
      s= []
      [s.append("{:}:{:}".format(k, edge[i][k])) for k in edge[i].keys()]
      s = "<src:" +src+ ", dst:" +dst+ ", " + ", ".join(s) + ">"
      edgeLabelDic[edge[i]['id']] = s
      print ("  ",i, ": ", s)

print("\n")


nx.write_graphml(G, graphml_filename, encoding='utf-8', prettyprint=True)




#####################################################################################
#  Initialize pydot graph and make pydot clusters
#####################################################################################

fsizeCluster0 = 80  # font size for central cluster
fsizeCluster1 = 50  # font size for sub cluster
margin=20

G2 = pydot.Dot(graph_type='digraph',fontname="Verdana",  title='rrr')
G2.set_node_defaults(fontsize='45', fontname="Sans", fixedsize=False, width=5, penwidth=3)
G2.set_edge_defaults(arrowsize='2') 


# clusters, org
cluster_orgs = pydot.Cluster('orgs', id='c01',
  label='<<BR/><B>Organizations</B>>', 
  bgcolor="gray99", margin=margin, penwidth=10, fontsize=fsizeCluster0)

cluster_orgs_member = pydot.Cluster('orgs_member', id='c02',
  label='<<BR/><B>Organizations<BR/>Member</B>>', 
  bgcolor="gray96", margin=margin, penwidth=5, fontsize=fsizeCluster1)

cluster_orgs_nonmember = pydot.Cluster('orgs_nonmember', id='c03',
  label='<<BR/><B>Organizations<BR/>Nonmember</B>>', 
  bgcolor="gray96", margin=margin, penwidth=5, fontsize=fsizeCluster1)


# clusters, persons, employed
cluster_persons_employed = pydot.Cluster('persons_employed', id='c04',
  label='<<BR/><B>Persons<BR/>Employed</B>>', 
  bgcolor="gray99", margin=margin, penwidth=10, fontsize=fsizeCluster0)

cluster_persons_member_employed = pydot.Cluster('persons_employed_member', id='c05',
  label='<<BR/><B>Member</B>>', 
  bgcolor="gray96", margin=margin, penwidth=5, fontsize=fsizeCluster1)

cluster_persons_nonmember_employed = pydot.Cluster('persons_employed_nonmember', id='c06',
  label='<<BR/><B>Nonmember</B>>', 
  bgcolor="gray96", margin=margin, penwidth=5, fontsize=fsizeCluster1)


# clusters, persons, unemployed
cluster_persons_unemployed = pydot.Cluster('persons_unemployed', id='c07',
  label='<<BR/><B>Persons<BR/>Unemployed</B>>', 
  bgcolor="gray99", margin=margin, penwidth=10, fontsize=fsizeCluster0)

cluster_persons_member_unemployed = pydot.Cluster('persons_unemployed_member', id='c08',
  label='<<BR/><B>Member</B>>', 
  bgcolor="gray96", margin=margin, penwidth=5, fontsize=fsizeCluster1)

cluster_persons_nonmember_unemployed = pydot.Cluster('persons_unemployed_nonmember', id='c09',
  label='<<BR/><B>Nonmember</B>>', 
  bgcolor="gray96", margin=margin, penwidth=5, fontsize=fsizeCluster1)


# clusters, CBFS
cluster_CBFS = pydot.Cluster('CBFS', id='c10',
  label='<<BR/><B>CBFS</B>>', 
  bgcolor="gray96", margin=margin, penwidth=10, fontsize=fsizeCluster0)


#####################################################################################
#  Make pydot nodes
#####################################################################################

# persons
personNodes = set(n for n in G.nodes_iter() if G.node[n]['kind']=='person')
for n in personNodes:
  if G.node[n]['member'] and G.node[n]['employed']:
    cluster_persons_member_employed.add_node(pydot.Node(n,label=n, id=G.node[n]['id']))

  if G.node[n]['member'] and not G.node[n]['employed']:
    cluster_persons_member_unemployed.add_node(pydot.Node(n,label=n, id=G.node[n]['id']))

  if not G.node[n]['member'] and G.node[n]['employed']:
    cluster_persons_nonmember_employed.add_node(pydot.Node(n,label=n, id=G.node[n]['id']))      

  if not G.node[n]['member'] and not G.node[n]['employed']:
    cluster_persons_nonmember_unemployed.add_node(pydot.Node(n,label=n, id=G.node[n]['id']))    


# orgs
orgNodes = set(n for n in G.nodes_iter() if G.node[n]['kind']=='org')
for n in orgNodes:
  if G.node[n]['member']:
    cluster_orgs_member.add_node(pydot.Node(n,label=n, id=G.node[n]['id']))
  else:
    cluster_orgs_nonmember.add_node(pydot.Node(n,label=n, id=G.node[n]['id']))
    

# CBFS
arms = set(n for n in G.nodes_iter() if G.node[n]['kind']=='CBFS')
for n in arms:
  cluster_CBFS.add_node(pydot.Node(n,label=n, id=G.node[n]['id']))
 

# RoC and Gov
#G2.add_node(pydot.Node('gov',label=n, id=G.node['gov']['id']))
#G2.add_node(pydot.Node('roc',label=n, id=G.node['roc']['id']))


# add subclusters to clusters and clusters to pydot graph
cluster_persons_employed.add_subgraph(cluster_persons_member_employed)
cluster_persons_employed.add_subgraph(cluster_persons_nonmember_employed)

cluster_persons_unemployed.add_subgraph(cluster_persons_member_unemployed)
cluster_persons_unemployed.add_subgraph(cluster_persons_nonmember_unemployed)

cluster_orgs.add_subgraph(cluster_orgs_member)
cluster_orgs.add_subgraph(cluster_orgs_nonmember)

G2.add_subgraph(cluster_orgs)
G2.add_subgraph(cluster_persons_employed)
G2.add_subgraph(cluster_persons_unemployed)
G2.add_subgraph(cluster_CBFS)


#####################################################################################
#  Make pydot edges
#####################################################################################

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
      if edge[i]['unit'] == 'tokens':
        G2.add_edge(pydot.Edge(src, dst, style='dashed', color='blue', penwidth="1", 
          id=edge[i]['id'], title=src + "-->" + dst, tooltip=edgeLabelDic[edge[i]['id']]))
      else:
        G2.add_edge(pydot.Edge(src, dst, style='solid', color='green', penwidth="1",  
          id=edge[i]['id'], title=src + "-->" + dst, tooltip=edgeLabelDic[edge[i]['id']]))
      
      

#####################################################################################
#  Print and save pydot results
#####################################################################################

# print clusters
print("\nSVG clusters:\n")

print("clusterDic = {}")
for c0 in G2.get_subgraphs():
  print("\nclusterDic['{}'] = {{'name':'{}', \n  'nodes':{}}}".format(
    c0.get_id(), c0.get_name(), str([node.get_name() for node in c0.get_nodes()])))
    
  for c1 in c0.get_subgraphs():
    print("clusterDic['{}'] = {{'name':'{}', \n  'nodes':{}}}".format(
      c1.get_id(), c1.get_name(), str([node.get_name() for node in c1.get_nodes()])))


print("\n")


#G2.set_esep(25)
G2.set_rankdir('LR')
#G2.set_overlap(False)
#G2.set_layout('fdp')
G2.set_splines(True)
G2.set_nodesep(3)
G2.set_ranksep(17)

G2.set_ratio(1.5)
G2.set_size(19)
G2.set_pad(.2)


G2.write_svg(svg_filename, prog='dot')

# copy files to destinations in leddaApp

shutil.copy(graphml_filename, '../leddaApp/static/data/')
shutil.copy(svg_filename, '../leddaApp/static/images/')


"""
To do:
<title>G</title>
<g id="graph00" class="svg-pan-zoom_viewport">

id= svg_detailed_graph
edge_token and edge_dollar classes

no titles on edges

fill="none"

not node18
"""





"""
Count of elements (nodes+edges) =  180

Nodes:
node:  CBFS_NP_donation    {'group': 'NP_donation', 'kind': 'CBFS', 'id': 16}
node:  CBFS_PB_subsidy    {'group': 'PB_subsidy', 'kind': 'CBFS', 'id': 15}
node:  CBFS_SB_subsidy    {'group': 'SB_subsidy', 'kind': 'CBFS', 'id': 14}
node:  CBFS_nurture    {'group': 'nurture', 'kind': 'CBFS', 'id': 17}
node:  gov    {'kind': 'gov', 'id': 18}
node:  org_member_NP    {'group': 'NP', 'kind': 'org', 'id': 9, 'member': True}
node:  org_member_PB    {'group': 'PB', 'kind': 'org', 'id': 11, 'member': True}
node:  org_member_SB    {'group': 'SB', 'kind': 'org', 'id': 10, 'member': True}
node:  org_nonmember_NP    {'group': 'NP', 'kind': 'org', 'id': 12, 'member': False}
node:  org_nonmember_SB    {'group': 'SB', 'kind': 'org', 'id': 13, 'member': False}
node:  person_member_NIWF    {'group': 'NIWF', 'employed': False, 'kind': 'person', 'id': 3, 'member': True}
node:  person_member_NP    {'group': 'NP', 'employed': True, 'kind': 'person', 'id': 0, 'member': True}
node:  person_member_PB    {'group': 'PB', 'employed': True, 'kind': 'person', 'id': 2, 'member': True}
node:  person_member_SB    {'group': 'SB', 'employed': True, 'kind': 'person', 'id': 1, 'member': True}
node:  person_member_unemployed    {'group': 'unemployed', 'employed': False, 'kind': 'person', 'id': 4, 'member': True}
node:  person_nonmember_NIWF    {'group': 'NIWF', 'employed': False, 'kind': 'person', 'id': 7, 'member': False}
node:  person_nonmember_NP    {'group': 'NP', 'employed': True, 'kind': 'person', 'id': 5, 'member': False}
node:  person_nonmember_SB    {'group': 'SB', 'employed': True, 'kind': 'person', 'id': 6, 'member': False}
node:  person_nonmember_unemployed    {'group': 'unemployed', 'employed': False, 'kind': 'person', 'id': 8, 'member': False}
node:  roc    {'kind': 'roc', 'id': 19}

Edges:

edge:  CBFS_NP_donation ----> org_member_NP
   0 :  <src:CBFS_NP_donation, dst:org_member_NP, unit:dollars, kind:CBFS_funding, id:167, value:0>
   1 :  <src:CBFS_NP_donation, dst:org_member_NP, unit:tokens, kind:CBFS_funding, id:168, value:0>

edge:  CBFS_PB_subsidy ----> org_member_PB
   0 :  <src:CBFS_PB_subsidy, dst:org_member_PB, unit:dollars, kind:CBFS_funding, id:165, value:0>
   1 :  <src:CBFS_PB_subsidy, dst:org_member_PB, unit:tokens, kind:CBFS_funding, id:166, value:0>

edge:  CBFS_SB_subsidy ----> org_member_SB
   0 :  <src:CBFS_SB_subsidy, dst:org_member_SB, unit:dollars, kind:CBFS_funding, id:163, value:0>
   1 :  <src:CBFS_SB_subsidy, dst:org_member_SB, unit:tokens, kind:CBFS_funding, id:164, value:0>

edge:  CBFS_nurture ----> person_member_NIWF
   0 :  <src:CBFS_nurture, dst:person_member_NIWF, unit:dollars, kind:nurture, id:137, value:0>
   1 :  <src:CBFS_nurture, dst:person_member_NIWF, unit:tokens, kind:nurture, id:138, value:0>

edge:  CBFS_nurture ----> person_member_unemployed
   0 :  <src:CBFS_nurture, dst:person_member_unemployed, unit:dollars, kind:nurture, id:127, value:0>
   1 :  <src:CBFS_nurture, dst:person_member_unemployed, unit:tokens, kind:nurture, id:128, value:0>

edge:  gov ----> org_member_NP
   0 :  <src:gov, dst:org_member_NP, unit:dollars, kind:gov_grants_contracts, id:171, value:0>

edge:  gov ----> org_member_PB
   0 :  <src:gov, dst:org_member_PB, unit:dollars, kind:gov_subsidies_contracts, id:170, value:0>

edge:  gov ----> org_member_SB
   0 :  <src:gov, dst:org_member_SB, unit:dollars, kind:gov_subsidies_contracts, id:169, value:0>

edge:  gov ----> org_nonmember_NP
   0 :  <src:gov, dst:org_nonmember_NP, unit:dollars, kind:gov_grants_contracts, id:173, value:0>

edge:  gov ----> org_nonmember_SB
   0 :  <src:gov, dst:org_nonmember_SB, unit:dollars, kind:gov_subsidies_contracts, id:172, value:0>

edge:  gov ----> person_member_NIWF
   0 :  <src:gov, dst:person_member_NIWF, unit:dollars, kind:gov_support, id:110, value:0>

edge:  gov ----> person_member_unemployed
   0 :  <src:gov, dst:person_member_unemployed, unit:dollars, kind:gov_support, id:108, value:0>

edge:  gov ----> person_nonmember_NIWF
   0 :  <src:gov, dst:person_nonmember_NIWF, unit:dollars, kind:gov_support, id:113, value:0>

edge:  gov ----> person_nonmember_unemployed
   0 :  <src:gov, dst:person_nonmember_unemployed, unit:dollars, kind:gov_support, id:117, value:0>

edge:  gov ----> roc
   0 :  <src:gov, dst:roc, unit:dollars, kind:gov_spending, id:174, value:0>

edge:  org_member_NP ----> person_member_NP
   0 :  <src:org_member_NP, dst:person_member_NP, unit:dollars, kind:wage, id:80, value:0>
   1 :  <src:org_member_NP, dst:person_member_NP, unit:tokens, kind:wage, id:81, value:0>

edge:  org_member_PB ----> person_member_PB
   0 :  <src:org_member_PB, dst:person_member_PB, unit:dollars, kind:wage, id:49, value:0>
   1 :  <src:org_member_PB, dst:person_member_PB, unit:tokens, kind:wage, id:50, value:0>

edge:  org_member_SB ----> person_member_SB
   0 :  <src:org_member_SB, dst:person_member_SB, unit:dollars, kind:wage, id:101, value:0>
   1 :  <src:org_member_SB, dst:person_member_SB, unit:tokens, kind:wage, id:102, value:0>

edge:  org_nonmember_NP ----> person_nonmember_NP
   0 :  <src:org_nonmember_NP, dst:person_nonmember_NP, unit:dollars, kind:wage, id:32, value:0>

edge:  org_nonmember_SB ----> person_nonmember_SB
   0 :  <src:org_nonmember_SB, dst:person_nonmember_SB, unit:dollars, kind:wage, id:24, value:0>

edge:  person_member_NIWF ----> CBFS_NP_donation
   0 :  <src:person_member_NIWF, dst:CBFS_NP_donation, unit:dollars, kind:contribution, id:143, value:0>
   1 :  <src:person_member_NIWF, dst:CBFS_NP_donation, unit:tokens, kind:contribution, id:144, value:0>

edge:  person_member_NIWF ----> CBFS_PB_subsidy
   0 :  <src:person_member_NIWF, dst:CBFS_PB_subsidy, unit:dollars, kind:contribution, id:141, value:0>
   1 :  <src:person_member_NIWF, dst:CBFS_PB_subsidy, unit:tokens, kind:contribution, id:142, value:0>

edge:  person_member_NIWF ----> CBFS_SB_subsidy
   0 :  <src:person_member_NIWF, dst:CBFS_SB_subsidy, unit:dollars, kind:contribution, id:139, value:0>
   1 :  <src:person_member_NIWF, dst:CBFS_SB_subsidy, unit:tokens, kind:contribution, id:140, value:0>

edge:  person_member_NIWF ----> CBFS_nurture
   0 :  <src:person_member_NIWF, dst:CBFS_nurture, unit:dollars, kind:contribution, id:145, value:0>
   1 :  <src:person_member_NIWF, dst:CBFS_nurture, unit:tokens, kind:contribution, id:146, value:0>

edge:  person_member_NIWF ----> gov
   0 :  <src:person_member_NIWF, dst:gov, unit:dollars, kind:taxes, id:111, value:0>

edge:  person_member_NIWF ----> org_member_NP
   0 :  <src:person_member_NIWF, dst:org_member_NP, unit:dollars, kind:spending, id:73, value:0>
   1 :  <src:person_member_NIWF, dst:org_member_NP, unit:tokens, kind:spending, id:74, value:0>
   2 :  <src:person_member_NIWF, dst:org_member_NP, unit:dollars, kind:regular_donation, id:75, value:0>

edge:  person_member_NIWF ----> org_member_PB
   0 :  <src:person_member_NIWF, dst:org_member_PB, unit:dollars, kind:spending, id:56, value:0>
   1 :  <src:person_member_NIWF, dst:org_member_PB, unit:tokens, kind:spending, id:57, value:0>

edge:  person_member_NIWF ----> org_member_SB
   0 :  <src:person_member_NIWF, dst:org_member_SB, unit:dollars, kind:spending, id:95, value:0>
   1 :  <src:person_member_NIWF, dst:org_member_SB, unit:tokens, kind:spending, id:96, value:0>

edge:  person_member_NIWF ----> org_nonmember_NP
   0 :  <src:person_member_NIWF, dst:org_nonmember_NP, unit:dollars, kind:spending, id:37, value:0>
   1 :  <src:person_member_NIWF, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:38, value:0>

edge:  person_member_NIWF ----> org_nonmember_SB
   0 :  <src:person_member_NIWF, dst:org_nonmember_SB, unit:dollars, kind:spending, id:23, value:0>

edge:  person_member_NP ----> CBFS_NP_donation
   0 :  <src:person_member_NP, dst:CBFS_NP_donation, unit:dollars, kind:contribution, id:151, value:0>
   1 :  <src:person_member_NP, dst:CBFS_NP_donation, unit:tokens, kind:contribution, id:152, value:0>

edge:  person_member_NP ----> CBFS_PB_subsidy
   0 :  <src:person_member_NP, dst:CBFS_PB_subsidy, unit:dollars, kind:contribution, id:149, value:0>
   1 :  <src:person_member_NP, dst:CBFS_PB_subsidy, unit:tokens, kind:contribution, id:150, value:0>

edge:  person_member_NP ----> CBFS_SB_subsidy
   0 :  <src:person_member_NP, dst:CBFS_SB_subsidy, unit:dollars, kind:contribution, id:147, value:0>
   1 :  <src:person_member_NP, dst:CBFS_SB_subsidy, unit:tokens, kind:contribution, id:148, value:0>

edge:  person_member_NP ----> CBFS_nurture
   0 :  <src:person_member_NP, dst:CBFS_nurture, unit:dollars, kind:contribution, id:153, value:0>
   1 :  <src:person_member_NP, dst:CBFS_nurture, unit:tokens, kind:contribution, id:154, value:0>

edge:  person_member_NP ----> gov
   0 :  <src:person_member_NP, dst:gov, unit:dollars, kind:taxes, id:115, value:0>

edge:  person_member_NP ----> org_member_NP
   0 :  <src:person_member_NP, dst:org_member_NP, unit:dollars, kind:spending, id:82, value:0>
   1 :  <src:person_member_NP, dst:org_member_NP, unit:tokens, kind:spending, id:83, value:0>
   2 :  <src:person_member_NP, dst:org_member_NP, unit:dollars, kind:regular_donation, id:84, value:0>

edge:  person_member_NP ----> org_member_PB
   0 :  <src:person_member_NP, dst:org_member_PB, unit:dollars, kind:spending, id:60, value:0>
   1 :  <src:person_member_NP, dst:org_member_PB, unit:tokens, kind:spending, id:61, value:0>

edge:  person_member_NP ----> org_member_SB
   0 :  <src:person_member_NP, dst:org_member_SB, unit:dollars, kind:spending, id:99, value:0>
   1 :  <src:person_member_NP, dst:org_member_SB, unit:tokens, kind:spending, id:100, value:0>

edge:  person_member_NP ----> org_nonmember_NP
   0 :  <src:person_member_NP, dst:org_nonmember_NP, unit:dollars, kind:spending, id:43, value:0>
   1 :  <src:person_member_NP, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:44, value:0>

edge:  person_member_NP ----> org_nonmember_SB
   0 :  <src:person_member_NP, dst:org_nonmember_SB, unit:dollars, kind:spending, id:27, value:0>

edge:  person_member_PB ----> CBFS_NP_donation
   0 :  <src:person_member_PB, dst:CBFS_NP_donation, unit:dollars, kind:contribution, id:123, value:0>
   1 :  <src:person_member_PB, dst:CBFS_NP_donation, unit:tokens, kind:contribution, id:124, value:0>

edge:  person_member_PB ----> CBFS_PB_subsidy
   0 :  <src:person_member_PB, dst:CBFS_PB_subsidy, unit:dollars, kind:contribution, id:121, value:0>
   1 :  <src:person_member_PB, dst:CBFS_PB_subsidy, unit:tokens, kind:contribution, id:122, value:0>

edge:  person_member_PB ----> CBFS_SB_subsidy
   0 :  <src:person_member_PB, dst:CBFS_SB_subsidy, unit:dollars, kind:contribution, id:119, value:0>
   1 :  <src:person_member_PB, dst:CBFS_SB_subsidy, unit:tokens, kind:contribution, id:120, value:0>

edge:  person_member_PB ----> CBFS_nurture
   0 :  <src:person_member_PB, dst:CBFS_nurture, unit:dollars, kind:contribution, id:125, value:0>
   1 :  <src:person_member_PB, dst:CBFS_nurture, unit:tokens, kind:contribution, id:126, value:0>

edge:  person_member_PB ----> gov
   0 :  <src:person_member_PB, dst:gov, unit:dollars, kind:taxes, id:106, value:0>

edge:  person_member_PB ----> org_member_NP
   0 :  <src:person_member_PB, dst:org_member_NP, unit:dollars, kind:spending, id:65, value:0>
   1 :  <src:person_member_PB, dst:org_member_NP, unit:tokens, kind:spending, id:66, value:0>
   2 :  <src:person_member_PB, dst:org_member_NP, unit:dollars, kind:regular_donation, id:67, value:0>

edge:  person_member_PB ----> org_member_PB
   0 :  <src:person_member_PB, dst:org_member_PB, unit:dollars, kind:spending, id:51, value:0>
   1 :  <src:person_member_PB, dst:org_member_PB, unit:tokens, kind:spending, id:52, value:0>

edge:  person_member_PB ----> org_member_SB
   0 :  <src:person_member_PB, dst:org_member_SB, unit:dollars, kind:spending, id:90, value:0>
   1 :  <src:person_member_PB, dst:org_member_SB, unit:tokens, kind:spending, id:91, value:0>

edge:  person_member_PB ----> org_nonmember_NP
   0 :  <src:person_member_PB, dst:org_nonmember_NP, unit:dollars, kind:spending, id:30, value:0>
   1 :  <src:person_member_PB, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:31, value:0>

edge:  person_member_PB ----> org_nonmember_SB
   0 :  <src:person_member_PB, dst:org_nonmember_SB, unit:dollars, kind:spending, id:20, value:0>

edge:  person_member_SB ----> CBFS_NP_donation
   0 :  <src:person_member_SB, dst:CBFS_NP_donation, unit:dollars, kind:contribution, id:159, value:0>
   1 :  <src:person_member_SB, dst:CBFS_NP_donation, unit:tokens, kind:contribution, id:160, value:0>

edge:  person_member_SB ----> CBFS_PB_subsidy
   0 :  <src:person_member_SB, dst:CBFS_PB_subsidy, unit:dollars, kind:contribution, id:157, value:0>
   1 :  <src:person_member_SB, dst:CBFS_PB_subsidy, unit:tokens, kind:contribution, id:158, value:0>

edge:  person_member_SB ----> CBFS_SB_subsidy
   0 :  <src:person_member_SB, dst:CBFS_SB_subsidy, unit:dollars, kind:contribution, id:155, value:0>
   1 :  <src:person_member_SB, dst:CBFS_SB_subsidy, unit:tokens, kind:contribution, id:156, value:0>

edge:  person_member_SB ----> CBFS_nurture
   0 :  <src:person_member_SB, dst:CBFS_nurture, unit:dollars, kind:contribution, id:161, value:0>
   1 :  <src:person_member_SB, dst:CBFS_nurture, unit:tokens, kind:contribution, id:162, value:0>

edge:  person_member_SB ----> gov
   0 :  <src:person_member_SB, dst:gov, unit:dollars, kind:taxes, id:116, value:0>

edge:  person_member_SB ----> org_member_NP
   0 :  <src:person_member_SB, dst:org_member_NP, unit:dollars, kind:spending, id:85, value:0>
   1 :  <src:person_member_SB, dst:org_member_NP, unit:tokens, kind:spending, id:86, value:0>
   2 :  <src:person_member_SB, dst:org_member_NP, unit:dollars, kind:regular_donation, id:87, value:0>

edge:  person_member_SB ----> org_member_PB
   0 :  <src:person_member_SB, dst:org_member_PB, unit:dollars, kind:spending, id:62, value:0>
   1 :  <src:person_member_SB, dst:org_member_PB, unit:tokens, kind:spending, id:63, value:0>

edge:  person_member_SB ----> org_member_SB
   0 :  <src:person_member_SB, dst:org_member_SB, unit:dollars, kind:spending, id:103, value:0>
   1 :  <src:person_member_SB, dst:org_member_SB, unit:tokens, kind:spending, id:104, value:0>

edge:  person_member_SB ----> org_nonmember_NP
   0 :  <src:person_member_SB, dst:org_nonmember_NP, unit:dollars, kind:spending, id:45, value:0>
   1 :  <src:person_member_SB, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:46, value:0>

edge:  person_member_SB ----> org_nonmember_SB
   0 :  <src:person_member_SB, dst:org_nonmember_SB, unit:dollars, kind:spending, id:28, value:0>

edge:  person_member_unemployed ----> CBFS_NP_donation
   0 :  <src:person_member_unemployed, dst:CBFS_NP_donation, unit:dollars, kind:contribution, id:133, value:0>
   1 :  <src:person_member_unemployed, dst:CBFS_NP_donation, unit:tokens, kind:contribution, id:134, value:0>

edge:  person_member_unemployed ----> CBFS_PB_subsidy
   0 :  <src:person_member_unemployed, dst:CBFS_PB_subsidy, unit:dollars, kind:contribution, id:131, value:0>
   1 :  <src:person_member_unemployed, dst:CBFS_PB_subsidy, unit:tokens, kind:contribution, id:132, value:0>

edge:  person_member_unemployed ----> CBFS_SB_subsidy
   0 :  <src:person_member_unemployed, dst:CBFS_SB_subsidy, unit:dollars, kind:contribution, id:129, value:0>
   1 :  <src:person_member_unemployed, dst:CBFS_SB_subsidy, unit:tokens, kind:contribution, id:130, value:0>

edge:  person_member_unemployed ----> CBFS_nurture
   0 :  <src:person_member_unemployed, dst:CBFS_nurture, unit:dollars, kind:contribution, id:135, value:0>
   1 :  <src:person_member_unemployed, dst:CBFS_nurture, unit:tokens, kind:contribution, id:136, value:0>

edge:  person_member_unemployed ----> gov
   0 :  <src:person_member_unemployed, dst:gov, unit:dollars, kind:taxes, id:109, value:0>

edge:  person_member_unemployed ----> org_member_NP
   0 :  <src:person_member_unemployed, dst:org_member_NP, unit:dollars, kind:spending, id:70, value:0>
   1 :  <src:person_member_unemployed, dst:org_member_NP, unit:tokens, kind:spending, id:71, value:0>
   2 :  <src:person_member_unemployed, dst:org_member_NP, unit:dollars, kind:regular_donation, id:72, value:0>

edge:  person_member_unemployed ----> org_member_PB
   0 :  <src:person_member_unemployed, dst:org_member_PB, unit:dollars, kind:spending, id:54, value:0>
   1 :  <src:person_member_unemployed, dst:org_member_PB, unit:tokens, kind:spending, id:55, value:0>

edge:  person_member_unemployed ----> org_member_SB
   0 :  <src:person_member_unemployed, dst:org_member_SB, unit:dollars, kind:spending, id:93, value:0>
   1 :  <src:person_member_unemployed, dst:org_member_SB, unit:tokens, kind:spending, id:94, value:0>

edge:  person_member_unemployed ----> org_nonmember_NP
   0 :  <src:person_member_unemployed, dst:org_nonmember_NP, unit:dollars, kind:spending, id:35, value:0>
   1 :  <src:person_member_unemployed, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:36, value:0>

edge:  person_member_unemployed ----> org_nonmember_SB
   0 :  <src:person_member_unemployed, dst:org_nonmember_SB, unit:dollars, kind:spending, id:22, value:0>

edge:  person_nonmember_NIWF ----> gov
   0 :  <src:person_nonmember_NIWF, dst:gov, unit:dollars, kind:taxes, id:114, value:0>

edge:  person_nonmember_NIWF ----> org_member_NP
   0 :  <src:person_nonmember_NIWF, dst:org_member_NP, unit:dollars, kind:spending, id:78, value:0>
   1 :  <src:person_nonmember_NIWF, dst:org_member_NP, unit:dollars, kind:regular_donation, id:79, value:0>

edge:  person_nonmember_NIWF ----> org_member_PB
   0 :  <src:person_nonmember_NIWF, dst:org_member_PB, unit:dollars, kind:spending, id:59, value:0>

edge:  person_nonmember_NIWF ----> org_member_SB
   0 :  <src:person_nonmember_NIWF, dst:org_member_SB, unit:dollars, kind:spending, id:98, value:0>

edge:  person_nonmember_NIWF ----> org_nonmember_NP
   0 :  <src:person_nonmember_NIWF, dst:org_nonmember_NP, unit:dollars, kind:spending, id:41, value:0>
   1 :  <src:person_nonmember_NIWF, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:42, value:0>

edge:  person_nonmember_NIWF ----> org_nonmember_SB
   0 :  <src:person_nonmember_NIWF, dst:org_nonmember_SB, unit:dollars, kind:spending, id:26, value:0>

edge:  person_nonmember_NP ----> gov
   0 :  <src:person_nonmember_NP, dst:gov, unit:dollars, kind:taxes, id:107, value:0>

edge:  person_nonmember_NP ----> org_member_NP
   0 :  <src:person_nonmember_NP, dst:org_member_NP, unit:dollars, kind:spending, id:68, value:0>
   1 :  <src:person_nonmember_NP, dst:org_member_NP, unit:dollars, kind:regular_donation, id:69, value:0>

edge:  person_nonmember_NP ----> org_member_PB
   0 :  <src:person_nonmember_NP, dst:org_member_PB, unit:dollars, kind:spending, id:53, value:0>

edge:  person_nonmember_NP ----> org_member_SB
   0 :  <src:person_nonmember_NP, dst:org_member_SB, unit:dollars, kind:spending, id:92, value:0>

edge:  person_nonmember_NP ----> org_nonmember_NP
   0 :  <src:person_nonmember_NP, dst:org_nonmember_NP, unit:dollars, kind:spending, id:33, value:0>
   1 :  <src:person_nonmember_NP, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:34, value:0>

edge:  person_nonmember_NP ----> org_nonmember_SB
   0 :  <src:person_nonmember_NP, dst:org_nonmember_SB, unit:dollars, kind:spending, id:21, value:0>

edge:  person_nonmember_SB ----> gov
   0 :  <src:person_nonmember_SB, dst:gov, unit:dollars, kind:taxes, id:112, value:0>

edge:  person_nonmember_SB ----> org_member_NP
   0 :  <src:person_nonmember_SB, dst:org_member_NP, unit:dollars, kind:spending, id:76, value:0>
   1 :  <src:person_nonmember_SB, dst:org_member_NP, unit:dollars, kind:regular_donation, id:77, value:0>

edge:  person_nonmember_SB ----> org_member_PB
   0 :  <src:person_nonmember_SB, dst:org_member_PB, unit:dollars, kind:spending, id:58, value:0>

edge:  person_nonmember_SB ----> org_member_SB
   0 :  <src:person_nonmember_SB, dst:org_member_SB, unit:dollars, kind:spending, id:97, value:0>

edge:  person_nonmember_SB ----> org_nonmember_NP
   0 :  <src:person_nonmember_SB, dst:org_nonmember_NP, unit:dollars, kind:spending, id:39, value:0>
   1 :  <src:person_nonmember_SB, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:40, value:0>

edge:  person_nonmember_SB ----> org_nonmember_SB
   0 :  <src:person_nonmember_SB, dst:org_nonmember_SB, unit:dollars, kind:spending, id:25, value:0>

edge:  person_nonmember_unemployed ----> gov
   0 :  <src:person_nonmember_unemployed, dst:gov, unit:dollars, kind:taxes, id:118, value:0>

edge:  person_nonmember_unemployed ----> org_member_NP
   0 :  <src:person_nonmember_unemployed, dst:org_member_NP, unit:dollars, kind:spending, id:88, value:0>
   1 :  <src:person_nonmember_unemployed, dst:org_member_NP, unit:dollars, kind:regular_donation, id:89, value:0>

edge:  person_nonmember_unemployed ----> org_member_PB
   0 :  <src:person_nonmember_unemployed, dst:org_member_PB, unit:dollars, kind:spending, id:64, value:0>

edge:  person_nonmember_unemployed ----> org_member_SB
   0 :  <src:person_nonmember_unemployed, dst:org_member_SB, unit:dollars, kind:spending, id:105, value:0>

edge:  person_nonmember_unemployed ----> org_nonmember_NP
   0 :  <src:person_nonmember_unemployed, dst:org_nonmember_NP, unit:dollars, kind:spending, id:47, value:0>
   1 :  <src:person_nonmember_unemployed, dst:org_nonmember_NP, unit:dollars, kind:regular_donation, id:48, value:0>

edge:  person_nonmember_unemployed ----> org_nonmember_SB
   0 :  <src:person_nonmember_unemployed, dst:org_nonmember_SB, unit:dollars, kind:spending, id:29, value:0>

edge:  roc ----> org_member_NP
   0 :  <src:roc, dst:org_member_NP, unit:dollars, kind:trade, id:177, value:0>

edge:  roc ----> org_member_PB
   0 :  <src:roc, dst:org_member_PB, unit:dollars, kind:trade, id:176, value:0>

edge:  roc ----> org_member_SB
   0 :  <src:roc, dst:org_member_SB, unit:dollars, kind:trade, id:175, value:0>

edge:  roc ----> org_nonmember_NP
   0 :  <src:roc, dst:org_nonmember_NP, unit:dollars, kind:trade, id:179, value:0>

edge:  roc ----> org_nonmember_SB
   0 :  <src:roc, dst:org_nonmember_SB, unit:dollars, kind:trade, id:178, value:0>



SVG clusters:

clusterDic = {}

clusterDic['c10'] = {'name':'cluster_CBFS', 
  'nodes':['CBFS_PB_subsidy', 'CBFS_SB_subsidy', 'CBFS_NP_donation', 'CBFS_nurture']}

clusterDic['c07'] = {'name':'cluster_persons_unemployed', 
  'nodes':[]}
clusterDic['c09'] = {'name':'cluster_persons_unemployed_nonmember', 
  'nodes':['person_nonmember_NIWF', 'person_nonmember_unemployed']}
clusterDic['c08'] = {'name':'cluster_persons_unemployed_member', 
  'nodes':['person_member_unemployed', 'person_member_NIWF']}

clusterDic['c04'] = {'name':'cluster_persons_employed', 
  'nodes':[]}
clusterDic['c05'] = {'name':'cluster_persons_employed_member', 
  'nodes':['person_member_NP', 'person_member_SB', 'person_member_PB']}
clusterDic['c06'] = {'name':'cluster_persons_employed_nonmember', 
  'nodes':['person_nonmember_SB', 'person_nonmember_NP']}

clusterDic['c01'] = {'name':'cluster_orgs', 
  'nodes':[]}
clusterDic['c03'] = {'name':'cluster_orgs_nonmember', 
  'nodes':['org_nonmember_SB', 'org_nonmember_NP']}
clusterDic['c02'] = {'name':'cluster_orgs_member', 
  'nodes':['org_member_PB', 'org_member_SB', 'org_member_NP']}

"""










