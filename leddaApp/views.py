

import os
import numpy as np

from flask import g, make_response, Markup
from flask import render_template, request, jsonify, send_from_directory
from flask import flash, redirect, url_for


from leddaApp import app

from leddaApp import setup_model
from leddaApp import fitness


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
  
  #for k in K:
  #  print(k, data[k])
  
  # remove lists, make floats, convert % to fractions
  _ = [data.__setitem__(k, float(data[k][0])/100.) for k in K if k[0:9] != 'variable_']

  # for checkboxes, convert on/off to integers 1/0
  for k in K:
    if k[0:9] == 'variable_':
      if data[k][0] == 'True':
        data[k] = 1
      else:
        data[k] = 0
      
  # fix nonfractions
  data['family_income_target_final'] = data['family_income_target_final'] * 100
  data['population'] = data['population'] * 100
  
  print("\ndata:")
  _ = [print("{}= {}".format(k, data[k])) for k in K]
  
  
  X, stocksDic, countsDic, histoDic = setup_model.setup(data)
  
  # get fitness scores
  fitnessDic, tableDic, summaryGraphDic = fitness.getFit(X, stocksDic, Print=False)
  
  return jsonify(result = {"msg":"OK", 'fitnessDic': fitnessDic, 
    'tableDic':tableDic, 'paramsDic':paramsDic, 'countsDic': countsDic, 'histoDic': histoDic,
    'summaryGraphDic': summaryGraphDic})
  
  









   
