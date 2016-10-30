from browser import document, alert, confirm, prompt, window, ajax, svg, html
from javascript import JSConstructor
import json
import math



##########################################################################################
# Load summary graph
##########################################################################################

def load_svg_summary(url="/static/images/steady_state_summary_2.svg"):
  req = ajax.ajax()
  req.bind('complete', load_svg_summary_complete)
  req.open('GET', url, False)
  req.send()


# ===========================================================
def load_svg_summary_complete(req):
  if req.status == 200 or req.status == 0:
    document['summary_graph_container'].html = req.text
  else:
    raise Exception()
    


##########################################################################################
# Load detailed graph
##########################################################################################

def load_svg_detailed(url="/static/images/steady_state_detailed.svg"):
  req = ajax.ajax()
  req.bind('complete', load_svg_detailed_complete)
  req.open('GET', url, False)
  req.send()


# ===========================================================
def load_svg_detailed_complete(req):
  if req.status == 200 or req.status == 0:
    document['graph_right'].html = req.text
  else:
    raise Exception()
    


##########################################################################################
# Pan Zoom 
##########################################################################################

def load_svg_pan_controls(url="/static/images/pan_controls_2.svg"):
  req = ajax.ajax()
  req.bind('complete', load_pan_controls_complete)
  req.open('GET', url, False)
  req.send()


# ===========================================================
def load_pan_controls_complete(req):
  if req.status == 200 or req.status == 0:
    document['graph_left'].html = req.text
  else:
    raise Exception()
    

# ===========================================================            
def set_pan_controls():
  """
  Set the pan/zoom controls and bindings
  """
  
  pan_detailed.setMinZoom(0.1)
  pan_detailed.setMaxZoom(10.0)
  pan_detailed.disableZoom()
  pan_detailed.disableControlIcons()
  pan_detailed.disableDblClickZoom()
  pan_detailed.disableMouseWheelZoom()
  pan_detailed.disablePan()
  pan_detailed.disableZoom()
  pan_detailed.fit();

  elt = document['svg_detailed_graph']
  elt.style.cursor = "default"
  elt.bind('click', doZoom)  
    
  for ID in ['pointer', 'zoom_in', 'zoom_out', 'pan', 'info', 'reset']:
    elt = document[ID]
    elt.bind('click', pan_zoom)
  
  window.bind('resize', resize_pan_graph)


# ===========================================================
def pan_zoom(evt):
  """
  Turn on/off pan, zoom, reset depending on which control icon is clicked. Set
  cursor accordingly.
  """

  #print("evt= ", evt.target.id, evt.currentTarget.id)
  
  elt = document['svg_detailed_graph']
  
  if evt.currentTarget.id == "pointer":
    elt.style.cursor = "default" 
    pan_detailed.disableMouseWheelZoom()
    pan_detailed.disableZoom()
    pan_detailed.disablePan()

  if evt.currentTarget.id == "zoom_in":
    elt.style.cursor = "zoom-in" 
    pan_detailed.enableMouseWheelZoom()
    pan_detailed.enableZoom()
    pan_detailed.disablePan()

  if evt.currentTarget.id == "zoom_out":
    elt.style.cursor = "zoom-out" 
    pan_detailed.enableMouseWheelZoom()
    pan_detailed.enableZoom()
    pan_detailed.disablePan()
  
  elif evt.currentTarget.id == "pan":
    elt.style.cursor = "move" 
    pan_detailed.disableMouseWheelZoom()
    pan_detailed.disableZoom()
    pan_detailed.enablePan()

  elif evt.currentTarget.id == "info":
    elt.style.cursor = "help" 
    pan_detailed.disableMouseWheelZoom()
    pan_detailed.disableZoom()
    pan_detailed.disablePan()

  elif evt.currentTarget.id == "reset":
    pan_detailed.reset()  
  
  
# ===========================================================
def doZoom(evt):
  """
  Zoom in or out, or none, depending on cursor state, if detailed graph is clicked.
  Code adapted from https://github.com/ariutta/svg-pan-zoom/issues/136
  """
  
  elt = document['svg_detailed_graph']
  if (elt.style.cursor != "zoom-in") and (elt.style.cursor != "zoom-out"):
    # do not do zoom
    return
  
  print("\ndoZoom:")
    
  # get transformed x,y point to zoom at
  pan = pan_detailed.getPan()
  sizes = pan_detailed.getSizes()
  zoom = sizes.realZoom
    
  pt = elt.createSVGPoint()
  pt.x = evt.clientX
  pt.y = evt.clientY
  pt = pt.matrixTransform(elt.getScreenCTM().inverse())
  x = pt.x
  y = pt.y

  #x = (x - pan.x) / zoom;
  #y = (y - pan.y)/zoom;
    
  #print("x= {}, y= {}".format(x,y))
  #print("evt: x= {}, y={}".format(evt.offsetX, evt.offsetY))

  if elt.style.cursor == "zoom-in":
    pan_detailed.zoomAtPointBy(1.2, {'x': x, 'y': y})
  if elt.style.cursor == "zoom-out":
    pan_detailed.zoomAtPointBy(.8333, {'x': x, 'y': y}) 
  
    
# ===========================================================
def resize_pan_graph(evt):
  """
  Resize the graph that has pan controls if window is resized
  """  
  print("\nresize pan graph:")

  pan_detailed.resize()
  pan_detailed.fit()
  pan_detailed.center()  



##########################################################################################
# Submit Model
##########################################################################################

def submit_model(evt):
  """
  run model
  """
  
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
    'person_member_spending_to_nonmember_SB_TS',
    'person_member_spending_to_nonmember_NP_pct',
    'person_member_spending_to_nonmember_NP_TS',
    'person_nonmember_spending_to_member_SB_pct',
    'person_nonmember_spending_to_member_NP_pct',
    'person_nonmember_spending_to_member_PB_pct',
    'person_nonmember_spending_to_nonmember_SB_pct',
    'person_nonmember_spending_to_nonmember_NP_pct',
    
    'variable_earmark_SB_subsidy',
    'variable_earmark_PB_subsidy',
    'variable_earmark_NP_donation',
    'variable_earmark_nurture',
    'variable_earmark_SB_subsidy_TS',
    'variable_earmark_PB_subsidy_TS',
    'variable_earmark_NP_donation_TS',
    'variable_earmark_nurture_TS',
    'variable_person_member_spending_to_member_SB_pct',
    'variable_person_member_spending_to_member_SB_TS',
    'variable_person_member_spending_to_member_NP_pct',
    'variable_person_member_spending_to_member_NP_TS',
    'variable_person_member_spending_to_member_PB_pct',
    'variable_person_member_spending_to_member_PB_TS',
    'variable_person_member_spending_to_nonmember_SB_pct',
    'variable_person_member_spending_to_nonmember_SB_TS',
    'variable_person_member_spending_to_nonmember_NP_pct',
    'variable_person_member_spending_to_nonmember_NP_TS',
    'variable_person_nonmember_spending_to_member_SB_pct',
    'variable_person_nonmember_spending_to_member_NP_pct',
    'variable_person_nonmember_spending_to_member_PB_pct',
    'variable_person_nonmember_spending_to_nonmember_SB_pct',
    'variable_person_nonmember_spending_to_nonmember_NP_pct'
    ]


  data = {}
  for w in widgets:
    if w[0:9] == 'variable_':
      data[w] = document[w].checked
    else:
      data[w] = document[w].value
      
  #send info  
  req = ajax.ajax()
  req.bind('complete', complete_submit)
  req.open('POST', 'http://localhost:5000/runModel', True)
  req.send(data)
  

# ===========================================================
def complete_submit(req):
  """
  Collect the results from the model run and then call functions to annotate graphs,
  write tables, etc.  
  """
  
  global fitnessDic
  global tableDic
  global paramsDic
  global countsDic
  global histoDic
  global summaryGraphDic
  
  print("\nreturned:")
  
  if req.status not in [200, 0]:
    raise Exception ("\n\nError in complete_submit\n")

  msg = json.loads(req.text)['result']['msg']
  if msg != 'OK':
    raise Exception ("\n\nError in complete_submit, msg not OK\n")
  
  fitnessDic = json.loads(req.text)['result']['fitnessDic']
  tableDic = json.loads(req.text)['result']['tableDic']
  paramsDic = json.loads(req.text)['result']['paramsDic']
  countsDic = json.loads(req.text)['result']['countsDic']
  histoDic = json.loads(req.text)['result']['histoDic']
  summaryGraphDic = json.loads(req.text)['result']['summaryGraphDic']
  
  # annotate graphs with returned data  
  annotate_detailed_graph_edges()
  annotate_detailed_graph_nodes()
  annotate_summary_graph()    
    
  # write out summary tables and highcharts figures for Group 1 and Group 2 parameters/variables
  write_group_1_summary_tables()
  write_group_2_summary_tables()
  
  # write out complete set of node tables
  write_all_tables()
  
  

##########################################################################################
# Color functions
##########################################################################################


def floatRgb(mag):
  """
  Return a tuple of floats between 0 and 1 for the red, green and
  blue amplitudes.
  floatRgb(mag, cmin, cmax)
  http://code.activestate.com/recipes/52273-colormap-returns-an-rgb-tuple-on-a-0-to-255-scale-/
  """
  #try:
  #  # normalize to [0,1]
  #  x = float(mag-cmin)/float(cmax-cmin)
  #except:
  #  # cmax = cmin
  #  x = 0.5
  
  #blue  = min((max((4*(0.75-x), 0.)), 1.))
  #red   = min((max((4*(x-0.25), 0.)), 1.))
  #green = min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
  
  green  = min((max((4*(0.75-mag), 0.)), 1.))
  red    = min((max((4*(mag-0.25), 0.)), 1.))
  blue   = min((max((4*math.fabs(mag-0.5)-1., 0.)), 1.))

  return (red, green, blue)

# ===========================================================
def strRgb(mag, cmin, cmax):
  """
  Return a tuple of strings to be used in Tk plots.
  """
  red, green, blue = floatRgb(mag, cmin, cmax)       
  return "#%02x%02x%02x" % (red*255, green*255, blue*255)


def rgb(mag):
  """
  Return a tuple of integers to be used in AWT/Java plots.
  rgb(mag, cmin, cmax)
  """
  red, green, blue = floatRgb(mag)
  return "rgba" + str((int(red*255), int(green*255), int(blue*255), .5))


def htmlRgb(mag, cmin, cmax):
  """
  Return a tuple of strings to be used in HTML documents.
  """
  return "#%02x%02x%02x"%rgb(mag, cmin, cmax)   



##########################################################################################
# Annotate Detailed Graph
##########################################################################################

def annotate_detailed_graph_edges():
  """
  Annotate edge weight and modal info for detailed graph
  """
  
  # edges
  # example: {20: {'unit': 'dollars', 'dst': 'org_member_SB', 'kind': 'spending', 
  #  'value': 10790695, 'id': 20, 'src': 'person_nonmember_SB'}
  
  print("\nannotate detailed graph edges:")
  
  global edgeDic
  edgeDic = {}
  
  edges = fitnessDic['edges']
  keys = list(edges.keys())
  keys.sort()
  minValue = 0
  maxValue = 0
  for i in keys:
    value = int(edges[i]['value'])  
    if value < minValue:
      minValue = value
    if value > maxValue:
      maxValue = value

  for i in keys:
    elt = document[i]
    elt.bind('click', showModal)

    table = html.TABLE(cellspacing=0, Class="infoTable")
    table <= html.CAPTION("Summary for Edge ID: {}".format(i))
    
    tb = html.TBODY()
    rows = []
    
    row = html.TR()
    row <= html.TD("Edge") + html.TD(edges[i]['src']+' &rarr; '+edges[i]['dst'])
    rows.append(row)

    row = html.TR()
    row <= html.TD("Kind") + html.TD(edges[i]['kind'])
    rows.append(row)

    value = int(edges[i]['value'])
    value2 = ((50-1)*(value-minValue)) / (maxValue-minValue) + 1 
    elt.set_style({'stroke-width': str(value2)+'px'})
    
    row = html.TR()
    row <= html.TD("Value") + html.TD("{:,} {:}".format(value, edges[i]['unit']))
    rows.append(row)
    
    tb <= rows
    table <= tb
    edgeDic[i] = table    


# ===========================================================  
def annotate_detailed_graph_nodes():
  """
  Annotate node color and modal info for detailed graph
  """
  
  print("\nannotate detailed graph nodes:")
  
  global nodeDic
  nodeDic = {}
  
  nodes = fitnessDic['nodes']
  keys = list(nodes.keys())
  keys.sort()
  minValue = 0
  maxValue = 0
  for i in keys:
    value = int(nodes[i]['fitness_total'])  
    if value < minValue:
      minValue = value
    if value > maxValue:
      maxValue = value

  # set maxValue to 1000 so that if all fitness values are tiny they do not look significant
  maxValue = max(maxValue, 1000)
  
  for i in keys:
    elt = document[i]
    elt.bind('click', showModal)
    
    # table 1: summary info ------------------------------------------------------
    table1 = html.TABLE(cellspacing=0, Class="infoTable")
    table1 <= html.CAPTION("Summary for Node ID: {}, Name: {}".format(i, nodes[i]['name']))
    
    tb = html.TBODY()
    rows = []
    
    row = html.TR()
    row <= html.TD("Kind") + html.TD(nodes[i]['kind'])
    rows.append(row)
    
    if 'member' in nodes[i].keys():
      row = html.TR()
      row <= html.TD("Member") + html.TD(nodes[i]['member'])
      rows.append(row)      

    if 'employed' in nodes[i].keys():
      row = html.TR()
      row <= html.TD("Employed") + html.TD(nodes[i]['employed'])
      rows.append(row) 
    
    if 'count' in nodes[i].keys():
      row = html.TR()
      row <= html.TD("Count") + html.TD(nodes[i]['count'])
      rows.append(row) 

    if 'actual_postCBFS_income_dollars_mean' in nodes[i].keys():
      row = html.TR()
      row <= html.TD("Mean post-CBFS Income") + html.TD(
        "{:,} dollars, {:,} tokens, {:,} T&D".format(
        nodes[i]['actual_postCBFS_income_dollars_mean'],
        nodes[i]['actual_postCBFS_income_tokens_mean'],
        nodes[i]['actual_postCBFS_income_total_mean']))
      rows.append(row)  

    if 'actual_preCBFS_income_dollars_mean' in nodes[i].keys():
      row = html.TR()
      row <= html.TD("Mean pre-CBFS Income") + html.TD(
        "{:,} dollars, {:,} tokens, {:,} T&D".format(
        nodes[i]['actual_preCBFS_income_dollars_mean'],
        nodes[i]['actual_preCBFS_income_tokens_mean'],
        nodes[i]['actual_preCBFS_income_total_mean']))
      rows.append(row)  

    row = html.TR()
    row <= html.TD("Fitness") + html.TD(
      "{:,} dollars, {:,} tokens, {:,} T&D".format(
      nodes[i]['fitness_dollars'],
      nodes[i]['fitness_tokens'],
      nodes[i]['fitness_total']))
    rows.append(row)  
        
    value = int(nodes[i]['fitness_total'])  
    value2 = ((1-0)*(value-0)) / (maxValue-0) + 0 
    
    # find ellipse element of parent
    ellipse = [c for c in elt.children if c.elt.nodeName == 'ellipse'][0]
    ellipse.set_style({'fill': rgb(value2) })
    
    tb <= rows
    table1 <= tb

    
    # tables 2 and 3: total flows and per person flows ------------------------
    inValues = tableDic[i]['in']['Values']
    inSums = tableDic[i]['in']['Sums']
    outValues = tableDic[i]['out']['Values']
    outSums = tableDic[i]['out']['Sums']    
    grandSums = tableDic[i]['grandSums']

    table2 = html.TABLE(cellspacing=0, Class="infoTable") # total flows
    table3 = html.TABLE(cellspacing=0, Class="infoTable") # per person flows
    
    for itable, table_ in enumerate([table2, table3]):
      if (itable == 1) and (nodes[i]['kind'] != 'person'):
        # only write table3 if node is a person
        continue
      
      if itable == 0:
        table_ <= html.CAPTION("Total Flows for Node ID: {}, Name: {}".format(i, nodes[i]['name']))
        count = 1 
      else:
        table_ <= html.CAPTION("Per Person Flows for Node ID: {}, Name: {}".format(i, nodes[i]['name'])) 
        count = nodes[i]['count']
      
      # write table body and rows
      tb = html.TBODY()
      rows = []
      
      # inflows heading row
      row = html.TR()
      row <= html.TH("In Flow Kind") + html.TH("Source") + \
        html.TH("Dollars") + html.TH("Tokens") + html.TH("T&D")
      rows.append(row)
          
      for v in inValues:
        row = html.TR()
        row <= html.TD(v[0]) + html.TD(v[1]) + \
          html.TD("{:,}".format(int(v[2]/count))) + \
          html.TD("{:,}".format(int(v[3]/count))) + \
          html.TD("{:,}".format(int(v[4]/count)))
        rows.append(row)      
      
      # subtotals
      v = inSums
      row = html.TR()
      row <= html.TH(v[0]) + html.TD(v[1]) + \
        html.TD("{:,}".format(int(v[2]/count))) + \
        html.TD("{:,}".format(int(v[3]/count))) + \
        html.TD("{:,}".format(int(v[4]/count)))
      rows.append(row)          

      # spacing row
      row = html.TR()
      row <= html.TD("") + html.TD("") + html.TD("") + html.TD("") + html.TD("")
      rows.append(row)
      
      # outflows heading row
      row = html.TR()
      row <= html.TH("Out Flow Kind") + html.TH("Destination") + \
        html.TH("Dollars") + html.TH("Tokens") + html.TH("T&D")
      rows.append(row)
          
      for v in outValues:
        row = html.TR()
        row <= html.TD(v[0]) + html.TD(v[1]) + \
          html.TD("{:,}".format(int(v[2]/count))) + \
          html.TD("{:,}".format(int(v[3]/count))) + \
          html.TD("{:,}".format(int(v[4]/count)))          
        rows.append(row)      
      
      # subtotals
      v = outSums
      row = html.TR()
      row <= html.TH(v[0]) + html.TD(v[1]) + \
        html.TD("{:,}".format(int(v[2]/count))) + \
        html.TD("{:,}".format(int(v[3]/count))) + \
        html.TD("{:,}".format(int(v[4]/count)))        
      rows.append(row)          

      # spacing row
      row = html.TR()
      row <= html.TD("") + html.TD("") + html.TD("") + html.TD("") + html.TD("")
      rows.append(row)

      # grand totals
      v = grandSums
      row = html.TR()
      row <= html.TH(v[0]) + html.TH(v[1]) + \
        html.TH("{:,}".format(int(v[2]/count))) + \
        html.TH("{:,}".format(int(v[3]/count))) + \
        html.TH("{:,}".format(int(v[4]/count)))      
      rows.append(row)  
          
      tb <= rows
      table_ <= tb
    
    if nodes[i]['kind'] == 'person': 
      nodeDic[i] = table1 + table2 + table3 
    else:
      nodeDic[i] = table1 + table2



# ===========================================================  
def annotate_summary_graph():
  """
  Annotate node color and modal info for detailed graph
  
  This function is unfinished.
  """
  
  print("\nannotate summary graph:")

  edges = summaryGraphDic['edges']
  print(edges)
  keys = list(edges.keys())
  keys.sort()
  minValue = 0
  maxValue = 0
  for i in keys:
    value = int(edges[i])  
    if value < minValue:
      minValue = value
    if value > maxValue:
      maxValue = value

  print('b', minValue, maxValue)
  for i in keys:
    elt = document[i]
    value = int(edges[i])
    value2 = ((5-1)*(value-minValue)) / (maxValue-minValue) + 1 
    elt.set_style({'stroke-width': str(value2)+'px'}) 
    elt.set_style({'stroke': 'red'})   







##########################################################################################
# Modal dialog box for graph annotation
##########################################################################################

def showModal(evt):
  """
  Show the model dialog for info/help clicks on detailed graph
  """
  
  print("\nshow Edge Table")
  
  elt = document['svg_detailed_graph']
  if elt.style.cursor != "help":
    # only show modal if cursor is 'help'
    return
  
  ID = evt.currentTarget.id  
  elt = document[ID]
  if ID in edgeDic.keys():
    dic = edgeDic
    document['modalContentContainer'].style.width = "300px"
  elif ID in nodeDic.keys():
    dic = nodeDic
    document['modalContentContainer'].style.width = "700px"
  else:
    raise Exception()
  
  table = dic[ID]
  
  
  modalContent = document['modalContent']
  modalContent <= table
  
  modal = document['modal_container']
  modal.style.display = 'block'
  

# ===========================================================  
def closeModal(evt):
  """
  Close the modal box if the x is clicked or somewhere outside the modal box
  is clicked
  """
  
  print("\ncloseModal:")
  
  modal = document['modal_container']
  close = document['closeModal']
  modalContent = document['modalContent']

  
  #print("evt.target == close: ", evt.target == close)
  #print("evt.target == modal: ", evt.target == modal)
  if (evt.target == close) or (evt.target == modal):
    #print("closing modal")
    modal.style.display = "none"
    modalContent.html = ' '



##########################################################################################
# Group 1 and Group 2 Summary tables and Highcharts figures
##########################################################################################

def write_group_1_summary_tables():
  """
  Write summary tables for Group 1 parameters
  """
  
  print("\nWrite groups 1 summary tables:")
  
  # Before/After
  table = html.TABLE(cellspacing=0, Class="summaryTable") 
  table <= html.CAPTION("Initial Vs. Final")
  population = float(paramsDic['population'][0])

  # write table body and rows
  tb = html.TBODY()
  rows = []

  row = html.TR()
  row <= html.TH("Item") + html.TH("Initial") + html.TH("Final")
  rows.append(row)

  # nodes
  nodes = fitnessDic['nodes']
  keys = [int(k) for k in nodes.keys()]
  keys.sort()
  for i in keys:
    i = str(i)
    if nodes[i]['kind'] != 'person':
      continue
    name = nodes[i]['name']
    row = html.TR()
    row <= html.TD('Count of ' + name) + \
      html.TD("{:,}".format(countsDic[name]['initial'])) +\
      html.TD("{:,}".format(countsDic[name]['final']))
    rows.append(row)  

  # spacing row
  row = html.TR()
  row <= html.TD("") + html.TD("") + html.TD("") 
  rows.append(row)  
  
  members = countsDic['memberCount']
  nonmembers = countsDic['nonmemberCount']
  total = members + nonmembers
  
  row = html.TR()
  row <= html.TD("Percent of population in membership") + \
    html.TD(0) +\
    html.TD("{:,}".format(round(members/total * 100, 1)))
  rows.append(row)    

  row = html.TR()
  row <= html.TD("Unemployment rate") + \
    html.TD(countsDic['unemploymemt_rate_initial']) +\
    html.TD(countsDic['unemploymemt_rate_final'])
  rows.append(row)  

  # percent of employed WF within population
  row = html.TR()
  row <= html.TD("Percent of employed workforce: SB") + \
    html.TD(countsDic['SB_pct_employed_initial']) +\
    html.TD(countsDic['SB_pct_employed_final'])
  rows.append(row)  

  row = html.TR()
  row <= html.TD("Percent of employed workforce: NP") + \
    html.TD(countsDic['NP_pct_employed_initial']) +\
    html.TD(countsDic['NP_pct_employed_final'])
  rows.append(row)  

  row = html.TR()
  row <= html.TD("Percent of employed workforce: PB") + \
    html.TD(countsDic['PB_pct_employed_initial']) +\
    html.TD(countsDic['PB_pct_employed_final'])
  rows.append(row)    

  # percent of employed WF within membership
  row = html.TR()
  row <= html.TD("Percent of employed membership workforce: SB") + \
    html.TD(0) +\
    html.TD(countsDic['SB_pct_employed_Ledda_final'])
  rows.append(row)  

  row = html.TR()
  row <= html.TD("Percent of employed membership workforce: NP") + \
    html.TD(0) +\
    html.TD(countsDic['NP_pct_employed_Ledda_final'])
  rows.append(row)  

  row = html.TR()
  row <= html.TD("Percent of employed membership workforce: PB") + \
    html.TD(0) +\
    html.TD(countsDic['PB_pct_employed_Ledda_final'])
  rows.append(row)  

  # mean incomes
  row = html.TR()
  row <= html.TD("Mean family income, T&D") + \
    html.TD("{:,}".format(2 * countsDic['mean_income_initial'])) +\
    html.TD("{:,}".format(2 * countsDic['mean_income_final']))
  rows.append(row)  

  row = html.TR()
  row <= html.TD("Mean family income, membership, T&D") + \
    html.TD(0) +\
    html.TD("{:,}".format(2 * countsDic['mean_income_final_Ledda']))
  rows.append(row)  

  row = html.TR()
  row <= html.TD("Mean family income, nonmembers") + \
    html.TD("{:,}".format(2 * countsDic['mean_income_initial'])) +\
    html.TD("{:,}".format(2 * countsDic['mean_income_final_nonLedda']))
  rows.append(row)  


  tb <= rows
  table <= tb  
  
  document['group_1_summary_table'].html = ' '
  document['group_1_summary_table'] <= table

  
  # ============== income histograms ===================================== 
  # set up histogram data, convert counts to percent, format bin edge data
  hist0 = histoDic['hist']
  bin_edges =  histoDic['bin_edges']  

  hist1 = hist0[:]
  for i, val in enumerate(histoDic['bin_edges']):
    if (i==0) or (i > len(histoDic['bin_edges'])-1):
      continue
    if (2 * countsDic['mean_income_final_Ledda']) > histoDic['bin_edges'][i]:
      hist1[i] += hist1[i-1]
      hist1[i-1] = 0
    else:
      break   
  
  hist0_sum = sum(hist0)
  hist1_sum = sum(hist1)
  hist0 = [round(v/hist0_sum * 100,1) for v in hist0]
  hist1 = [round(v/hist1_sum * 100,1) for v in hist1]
  bin_edges = ["{:,}".format(int(e)) for e in bin_edges]

  
  # ---------- make Highcharts code for income histograms ------- 
  colors = ['red', 'blue', 'orange', 'green']
  
  # create population distribution plot 
  title = {"text": "Family Income Distribution"}
  
  yAxis = [{ 
    # Primary yAxis, population %
    'title': {
      'text': 'Percent of Population',
      'style': {'color': 'black'}
      },
    'labels': {
      'style': {'color': 'black'}}
      }]
  
  xAxis = {
    "title": {"text": 'Family Income (T&D)'},
    'categories': bin_edges,
    'tickInterval': 10}

  legend = {}
  tooltip = {'shared': True}
  chartID = 'income_dist'
  chart_height = 500
  chart_width = 800

  chart = {"renderTo":chartID, "height":chart_height, "width":chart_width}
  
  plotOptions = {
    'column': {
      'pointPadding': 0,
      'borderWidth': 0,
      'groupPadding': 0,
      'shadow': False
      }}
  
  # create data series
  dataSeries = [{
    'name': 'Initial',
    'type': 'column',
    'yAxis': 0,
    'color': colors[1],
    'data': hist0
    }, {
    'name': 'Final',
    'type': 'column',
    'yAxis': 0,
    'color': colors[2],
    'data': hist1
    }]

  plt = {'chartID':chartID, 'chart':chart, 'series':dataSeries, 'title':title, 
    'xAxis':xAxis, 'yAxis':yAxis, 'legend':legend, 'plotOptions':plotOptions, 'tooltip':tooltip}

  b_highchart = JSConstructor(window.Highcharts.Chart)
  b_highchart(plt)


# =========================================================== 
def write_group_2_summary_tables():
  """
  Write summary tables for Group 2 parameters/variables
  """
  
  print("\nWrite groups 2 summary tables:")
  
  # total fitness
  table = html.TABLE(cellspacing=0, Class="summaryTable") 
  table <= html.CAPTION("Total Fitness")
  population = float(paramsDic['population'][0])

  # write table body and rows
  tb = html.TBODY()
  rows = []

  row = html.TR()
  row <= html.TH("Unit") + html.TH("Dollars") + html.TH("Tokens") + html.TH("T&D")
  rows.append(row)
  
  row = html.TR()
  row <= html.TD("Grand Total") + \
    html.TD("{:,}".format(fitnessDic['fitness']['dollars'])) + \
    html.TD("{:,}".format(fitnessDic['fitness']['tokens'] )) + \
    html.TD("{:,}".format(fitnessDic['fitness']['total']  )) 
  rows.append(row)
  
  row = html.TR()
  row <= html.TD("Log (Grand Total+1)") + \
    html.TD("{:,}".format(round(math.log(fitnessDic['fitness']['dollars']+1),2))) + \
    html.TD("{:,}".format(round(math.log(fitnessDic['fitness']['tokens'] +1),2))) + \
    html.TD("{:,}".format(round(math.log(fitnessDic['fitness']['total']  +1),2))) 
  rows.append(row)
  
  row = html.TR()
  row <= html.TD("Per Person Basis") + \
    html.TD("{:,}".format(int(round(fitnessDic['fitness']['dollars']/population)))) + \
    html.TD("{:,}".format(int(round(fitnessDic['fitness']['tokens'] /population)))) + \
    html.TD("{:,}".format(int(round(fitnessDic['fitness']['total']  /population)))) 
  rows.append(row)
  
  tb <= rows
  table <= tb  
  
  document['group_2_total_fitness_table'].html = ' '
  document['group_2_total_fitness_table'] <= table
  

  # group 2 node table
  table = html.TABLE(cellspacing=0, Class="summaryTable") 
  table <= html.CAPTION("Node Flows")

  tb = html.TBODY()
  rows = []

  row = html.TR()
  row <= html.TH("Node") + html.TH("Flow") + html.TH("Dollars") + \
    html.TH("Tokens") + html.TH("T&D")
  rows.append(row)

  nodes = fitnessDic['nodes']
  keys = [int(k) for k in nodes.keys()]
  keys.sort()
  for i in keys:
    i = str(i)

    inSums = tableDic[i]['in']['Sums']
    outSums = tableDic[i]['out']['Sums']    
    grandSums = tableDic[i]['grandSums']

    name = nodes[i]['name']
    row = html.TR()
    v = inSums
    row <= html.TD(name, rowspan="4") + html.TD("Inflows") + \
      html.TD("{:,}".format(v[2])) + \
      html.TD("{:,}".format(v[3])) + \
      html.TD("{:,}".format(v[4]))
    rows.append(row)  

    row = html.TR()
    v = outSums
    row <= html.TD("Outflows") + \
      html.TD("{:,}".format(v[2])) + \
      html.TD("{:,}".format(v[3])) + \
      html.TD("{:,}".format(v[4]))
    rows.append(row)  

    row = html.TR()
    v = grandSums
    row <= html.TD("Sum Flows") + \
      html.TD("{:,}".format(v[2])) + \
      html.TD("{:,}".format(v[3])) + \
      html.TD("{:,}".format(v[4]))
    rows.append(row)      

    row = html.TR()
    v = grandSums
    row <= html.TH("Fitness") + \
      html.TD("{:,}".format(nodes[i]['fitness_dollars'])) + \
      html.TD("{:,}".format(nodes[i]['fitness_tokens'])) + \
      html.TD("{:,}".format(nodes[i]['fitness_total']))
    rows.append(row)    

    # spacing row
    row = html.TR()
    row <= html.TD("") + html.TD("") + html.TD("") + html.TD("") + html.TD("") 
    rows.append(row)  


  tb <= rows
  table <= tb  
  
  document['group_2_node_table'].html = ' '
  document['group_2_node_table'] <= table



# =========================================================== 
def write_all_tables():
  """
  Write all node tables out 
  """
  
  print("\nwrite all tables:")
  
  elt = document['all_tables']
  elt.html = " "
  
  # nodes
  nodes = fitnessDic['nodes']
  keys = [int(k) for k in nodes.keys()]
  keys.sort()
  for i in keys:
    i = str(i)
    elt <= nodeDic[i]
    elt <= html.HR()
     
  if 1==2:
    # edges (this info is also contained in node tables)
    elt <= html.H3("Detailed edge tables")
    edges = fitnessDic['edges']
    keys = [int(k) for k in edges.keys()]
    keys.sort()
    for i in keys:
      i = str(i)
      elt <= edgeDic[i]    





##########################################################################################
# Bindings, etc on load
##########################################################################################

window.bind('click', closeModal)

load_svg_summary()
load_svg_detailed()
load_svg_pan_controls()

pan_detailed = window.svgPanZoom("#svg_detailed_graph")
set_pan_controls()

edgeDic = None
nodeDic = None
fitnessDic = None
tableDic = None
paramsDic = None
countsDic = None
histoDic = None
summaryGraphDic = None

close = document['closeModal']
close.bind('click', closeModal)
document['submit'].bind('click', submit_model)









