

import os
import numpy as np

from flask import g, make_response, Markup
from flask import render_template, request, jsonify, send_from_directory
from flask import flash, redirect, url_for


from leddaApp import app

from leddaApp import setup_model
from leddaApp import fitness
from leddaApp import optimizer

app.config.from_envvar('leddaApp_SETTINGS', silent=True)


# Load default config and override config from an environment variable
app.config.update(dict(
    DEBUG=True,
    SECRET_KEY = 'development key',

    PROJECT_FOLDER = os.path.join(os.path.dirname(app.root_path), 'Projects'),
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'data')

    ))

app.secret_key = app.config['SECRET_KEY']



#####################################################################
#               Index Page
#####################################################################

@app.route('/')
@app.route('/index')
def index():
  """
  This is the home page, where user lands first. 
  """
  return render_template('index.html')  



#####################################################################
#               Steady State
#####################################################################
@app.route('/steady_01')
def steady_01():
  """
  This is the steady_01 page. 
  """
  return render_template('steady_01.html')  


# -------------------------------------------------------------------------------------
@app.route('/runModel', methods=['POST'])
def runModel():
  """
  Run SS model 
  """
  
  data = dict(request.form)
  
  K = list(data.keys())
  K.sort()
  paramsDic = data.copy()
  
  print("initial data:")
  for k in K:
    print(k, data[k])
  
  
  # remove lists, make floats, convert % to fractions
  _ = [data.__setitem__(k, float(data[k][0])/100.) \
    for k in K if k[0:9] not in ['flexible_', 'doOptimiz']]
  
  # for checkboxes, convert on/off to integers 1/0
  for k in K:
    if k[0:9] in ['variable_', 'doOptimiz']:
      if data[k][0] == 'true':
        data[k] = 1
      else:
        data[k] = 0

  # fix nonfractions
  data['family_income_target_final'] = data['family_income_target_final'] * 100
  data['population'] = data['population'] * 100
  
  print("\ndata:")
  _ = [print("{}= {}".format(k, data[k])) for k in K]
  
  
  X, stocksDic, countsDic, histoDic = setup_model.setup(data)
  
  # delete items no long necessary
  del X.TP
  del X.TF
  
  if X.doOptimization == 0:
    # just run the fitness function once and return
    fitnessDic, tableDic, summaryGraphDic = fitness.getFit(X, stocksDic, Print=False, Optimize=False)
  else:
    # call optimizer (genetic algorithm)
    fitnessDic, tableDic, summaryGraphDic = optimizer.genetic(X, stocksDic)    
    
  resultDic = {
    "msg":"OK", 
    'fitnessDic': fitnessDic, 
    'tableDic':tableDic, 
    'paramsDic':paramsDic, 
    'countsDic': countsDic, 
    'histoDic': histoDic,
    'summaryGraphDic': summaryGraphDic
    }
  
  return render_template('results.html', results=resultDic) 
  

  
  









   
