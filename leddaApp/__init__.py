
from flask import Flask
#from flask.ext.cache import Cache 
from flask_mail import Mail

app = Flask(__name__)
mail = Mail(app)
#app.config['CACHE_TYPE'] = 'simple'

# register the cache instance and bind it to app 
#app.cache = Cache(app) 

import leddaApp.views





