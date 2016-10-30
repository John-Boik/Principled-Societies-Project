from browser import document, alert, confirm, prompt, window, ajax, svg, html
import json
import math


##########################################################################################
# Summary Graph
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
# Detailed Graph
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
# Pan Zoom Controls
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
    
            
    
##########################################################################################
# Pan-Zoom
##########################################################################################

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
  # set default cursor to default
  elt.style.cursor = "default"
  elt.bind('click', doZoom)  
    
  for ID in ['pointer', 'zoom_in', 'zoom_out', 'pan', 'info', 'reset']:
    elt = document[ID]
    elt.bind('click', pan_zoom)
  

# ===========================================================
def pan_zoom(evt):
  """
  Pan, zoom, reset
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
  Zoom in or out, or none, depending on cursor state.
  Code adapted from https://github.com/ariutta/svg-pan-zoom/issues/136
  """
  
  #print("\ndoZoom\n")
  
  elt = document['svg_detailed_graph']
  
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
    pan_detailed.zoomAtPointBy(.8, {'x': x, 'y': y}) 
  
    
  
    

##########################################################################################
# Submit Model
##########################################################################################

def submit_model(evt):
  """
  run model
  """
  
  widgets = [
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
  results of model run
  """
  global fitnessDic
  global tableDic
  
  print("\nreturned:")
  
  if req.status not in [200, 0]:
    raise Exception ("\n\nError in complete_submit\n")

  msg = json.loads(req.text)['result']['msg']
  if msg != 'OK':
    raise Exception ("\n\nError in complete_submit, msg not OK\n")
  
  fitnessDic = json.loads(req.text)['result']['fitnessDic']
  tableDic = json.loads(req.text)['result']['tableDic']
  
  annotate_detailed_graph_edges()
  annotate_detailed_graph_nodes()
  
  # todo: adjust_summary_graph(fitnessDic, tableDic)    
    



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
    elt.bind('click', showInfoTable)

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
  
  """
  node example:
  2: {'fitness_dollars': 21865596, 'postCBFS_income_total': 108201500.0, 'member': True, 
  'employed': True, 'actual_preCBFS_income_total_mean': 145256, 'group': 'PB', 
  'actual_preCBFS_income_dollars_mean': 94417, 'postCBFS_income_mean': 56650.0, 
  'actual_postCBFS_income_dollars_mean': 36822, 'count': 1910, 
  'actual_postCBFS_income_tokens_mean': 19827, 'kind': 'person', 
  'actual_postCBFS_income_total_mean': 56650, 'fitness_tokens': 21865596, 
  'actual_preCBFS_income_tokens_mean': 50840, 'name': 'person_member_PB', 
  'fitness_total': 0, 'id': 2}
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

  for i in keys:
    elt = document[i]
    elt.bind('click', showInfoTable)
    
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

    
    # table 2: total flows ---------------------------------------------------------
    inValues = tableDic[i]['in']['Values']
    inSums = tableDic[i]['in']['Sums']
    outValues = tableDic[i]['out']['Values']
    outSums = tableDic[i]['out']['Sums']    
    grandSums = tableDic[i]['grandSums']
    
    table2 = html.TABLE(cellspacing=0, Class="infoTable")
    table2 <= html.CAPTION("Total Flows for Node ID: {}, Name: {}".format(i, nodes[i]['name'])) 
    
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
        html.TD("{:,}".format(v[2])) + html.TD("{:,}".format(v[3])) + html.TD("{:,}".format(v[4]))
      rows.append(row)      
    
    # subtotals
    v = inSums
    row = html.TR()
    row <= html.TH(v[0]) + html.TD(v[1]) + \
      html.TD("{:,}".format(v[2])) + html.TD("{:,}".format(v[3])) + html.TD("{:,}".format(v[4]))
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
        html.TD("{:,}".format(v[2])) + html.TD("{:,}".format(v[3])) + html.TD("{:,}".format(v[4]))
      rows.append(row)      
    
    # subtotals
    v = outSums
    row = html.TR()
    row <= html.TH(v[0]) + html.TD(v[1]) + \
      html.TD("{:,}".format(v[2])) + html.TD("{:,}".format(v[3])) + html.TD("{:,}".format(v[4]))
    rows.append(row)          

    # spacing row
    row = html.TR()
    row <= html.TD("") + html.TD("") + html.TD("") + html.TD("") + html.TD("")
    rows.append(row)

    # grand totals
    v = grandSums
    row = html.TR()
    row <= html.TH(v[0]) + html.TH(v[1]) + \
      html.TH("{:,}".format(v[2])) + html.TH("{:,}".format(v[3])) + html.TH("{:,}".format(v[4]))
    rows.append(row)  
        
    tb <= rows
    table2 <= tb

    
    # table 3: per person flows for persons -------------------------------------------
    if nodes[i]['kind'] == 'person':
      count = nodes[i]['count']
      table3 = html.TABLE(cellspacing=0, Class="infoTable")
      table3 <= html.CAPTION("Per Person Flows for Node ID: {}, Name: {}".format(i, nodes[i]['name'])) 
      
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
      table3 <= tb
    
    if nodes[i]['kind'] == 'person': 
      nodeDic[i] = table1 + table2 + table3 
    else:
      nodeDic[i] = table1 + table2

  write_summary_tables()
  
  write_all_tables()
  


# ===========================================================
def showInfoTable(evt):
  """
  Show model dialog for edge table on edge click
  """
  
  print("\nshow Edge Table")
    
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




# =========================================================== 
def write_summary_tables()
  """
  Write summary tables out
  """
  
  # total fitness
  










# =========================================================== 
def write_all_tables():
  """
  Write all node tables out 
  """
  
  print("\nwrite all tables:")
  
  elt = document['all_tables']
  
  # nodes
  nodes = fitnessDic['nodes']
  keys = [int(k) for k in nodes.keys()]
  print (keys)
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

close = document['closeModal']
close.bind('click', closeModal)
document['submit'].bind('click', submit_model)







"""



test.node:
['Class', '__add__', '__bases__', '__bool__', '__class__', '__contains__', '__del__', '__delattr__', '__delitem__', '__dir__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__name__', '__ne__', '__new__', '__next__', '__or__', '__radd__', '__repr__', '__setattr__', '__setitem__', '__str__', '__subclasshook__', 'abs_left', 'abs_top', 'bind', 'children', 'class_name', 'clear', 'clone', 'closest', 'elt', 'events', 'focus', 'get', 'getContext', 'getSelectionRange', 'height', 'html', 'id', 'inside', 'left', 'options', 'parent', 'query', 'remove', 'reset', 'setSelectionRange', 'set_class_name', 'set_html', 'set_style', 'set_text', 'set_value', 'style', 'submit', 'text', 'toString', 'top', 'trigger', 'unbind', 'value', 'width']


pan-zoom:
['center', 'contain', 'destroy', 'disableControlIcons', 'disableDblClickZoom', 'disableMouseWheelZoom', 'disablePan', 'disableZoom', 'enableControlIcons', 'enableDblClickZoom', 'enableMouseWheelZoom', 'enablePan', 'enableZoom', 'fit', 'getPan', 'getSizes', 'getZoom', 'isControlIconsEnabled', 'isDblClickZoomEnabled', 'isMouseWheelZoomEnabled', 'isPanEnabled', 'isZoomEnabled', 'pan', 'panBy', 'reset', 'resetPan', 'resetZoom', 'resize', 'setBeforePan', 'setBeforeZoom', 'setMaxZoom', 'setMinZoom', 'setOnPan', 'setOnZoom', 'setZoomScaleSensitivity', 'updateBBox', 'zoom', 'zoomAtPoint', 'zoomAtPointBy', 'zoomBy', 'zoomIn', 'zoomOut']
https://github.com/ariutta/svg-pan-zoom

['0', 'context', 'length', 'selector']
"""





