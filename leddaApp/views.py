

import os, re
import numpy as np

from flask import g, make_response, Markup
from flask import render_template, request, jsonify, send_from_directory
from flask import flash, redirect, url_for
from flask_mail import Message


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
  DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'data'),
  MAIL_SERVER = 'smtp.gmail.com',
  MAIL_PORT = 465,
  MAIL_USE_SSL = True,
  MAIL_USERNAME = 'info@PrincipledSocietiesProject.org',
  MAIL_PASSWORD = 'yourMailPassword'

  ))



#######################################################################################
#               Serve basic pages
#######################################################################################

@app.route('/')
@app.route('/index')
def index():
  """
  This is the home page, where visitor lands first. 
  """
  return render_template('index.html')  

# -------------------------------------------------------------------------------------
@app.route('/survey')
def survey():
  """
  This is survey page 
  """
  return render_template('survey.html')  

# -------------------------------------------------------------------------------------
@app.route('/who_should_be_interested')
def who_should_be_interested():
  """
  This is who should be interested page
  """
  return render_template('who_should_be_interested.html')  


# -------------------------------------------------------------------------------------
@app.route('/model_steady_state_01')
def model_steady_state_01():
  """
  This is the steady_01 page. 
  """
  return render_template('model_steady_state_01.html')  


# -------------------------------------------------------------------------------------
@app.route('/glossary')
def glossary():
  """
  This is the glossary page. 
  """
  return render_template('glossary.html')  

  
# -------------------------------------------------------------------------------------
@app.route('/income_generation')
def income_generation():
  """
  This page is about income distributions are generated for the models. 
  """
  return render_template('income_generation.html') 


# -------------------------------------------------------------------------------------
@app.route('/about_psp')
def about_psp():
  """
  This page is about PSP. 
  """
  return render_template('about_psp.html') 


# -------------------------------------------------------------------------------------
@app.route('/contact')
def contact():
  """
  This is the contact page. 
  """
  return render_template('contact.html') 


# -------------------------------------------------------------------------------------
@app.route('/donate')
def donate():
  """
  This is the donate page. 
  """
  return render_template('donate.html') 


# -------------------------------------------------------------------------------------
@app.route('/articles_media')
def articles_media():
  """
  This is the income generation page. 
  """
  return render_template('articles_media.html') 


# -------------------------------------------------------------------------------------
@app.route('/book_edd_about')
def book_edd_about():
  """
  This page is about the book Economic Direct Democracy. 
  """
  return render_template('book_edd_about.html') 


# -------------------------------------------------------------------------------------
@app.route('/book_edd_download')
def book_edd_download():
  """
  This is the download page for Economic Direct Democracy. 
  """
  return render_template('book_edd_download.html') 
  
  
# -------------------------------------------------------------------------------------
@app.route('/book_edd_license')
def book_edd_license():
  """
  This page is about the creative commons license for Economic Direct Democracy. 
  """
  return render_template('book_edd_license.html')   


# -------------------------------------------------------------------------------------
@app.route('/book_edd_praise')
def book_edd_praise():
  """
  This page is about the creative commons license for Economic Direct Democracy. 
  """
  return render_template('book_edd_praise.html')   
  

# -------------------------------------------------------------------------------------
@app.route('/book_edd_toc')
def book_edd_toc():
  """
  This is the income generation page. 
  """
  return render_template('book_edd_toc.html') 


# -------------------------------------------------------------------------------------
@app.route('/contact_form', methods=['POST'])
def contact_form():
  """
  This mails the contact form. 
  """
  
  form = dict(request.form)
  
  if not re.match(r"[^@]+@[^@]+\.[^@]+", form['email'][0]):
    return jsonify(msg="Email validation fails")
  
  match = re.match(
    "facebook|twitter|instagram|youtube|you tube|design|website|follower|fan|visitor|roi",
    form['message'][0].lower())

  print("message: ", match)
  if match:
    return jsonify(msg="Message validation fails")
        
  msg = Message("PSP Message", sender= 'info@PrincipledSocietiesProject.org', 
    recipients=['info@PrincipledSocietiesProject.org'])
  msg.body = ("""
    From: {:} <{:}>,
    {:}
    """).format(form['name'][0], form['email'][0], form['message'][0])

  if (form['magic'][0] in ['4', 'four']) and (not match) and (form['other']==""):  
    #mail.send(msg)
    pass
    
  print(str(msg))
  
  return jsonify(msg="OK")


  
#######################################################################################
# Run the steady state model
#######################################################################################

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
    for k in K if ((k[0:9] not in ['flexible_']) and (k not in ['doGenetic', 'doBFGS','doRandomStart']))]
  
  # for checkboxes, convert on/off to integers 1/0
  for k in K:
    if (k[0:9] in ['flexible_']) or (k in ['doGenetic', 'doBFGS','doRandomStart']):
      if data[k][0] == 'true':
        data[k] = 1
      else:
        data[k] = 0

  if (data['doGenetic'] or data['doBFGS']):
    data['doOptimization'] = 1
  else:
    data['doOptimization'] = 0
  
  # fix nonfractions
  data['family_income_target_final'] = data['family_income_target_final'] * 100
  data['population'] = data['population'] * 100
  
  #print("\ndata:")
  #_ = [print("{}= {}".format(k, data[k])) for k in K]
  
  
  X, stocksDic, countsDic, histoDic = setup_model.setup(data)
  
  # delete attributes of X that are no long necessary
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
  

  
  









   
