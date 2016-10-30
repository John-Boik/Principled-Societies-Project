
from flask import Flask
#from flask.ext.cache import Cache 


app = Flask(__name__)
#app.config['CACHE_TYPE'] = 'simple'

# register the cache instance and bind it to app 
#app.cache = Cache(app) 

import leddaApp.views





