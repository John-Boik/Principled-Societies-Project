

import os, re
import numpy as np

from flask import g, make_response, Markup
from flask import render_template, request, jsonify, send_from_directory
from flask import flash, redirect, url_for
from flask_mail import Message

from leddaApp import app

from flask_mail import Mail


from leddaApp import setup_model
from leddaApp import fitness
from leddaApp import optimizer

from leddaApp.secrets import MAIL_SERVER, MAIL_PASSWORD, MAIL_USERNAME, MAIL_PORT, MAIL_USE_SSL

app.config.from_envvar('leddaApp_SETTINGS', silent=True)


# Load default config and override config from an environment variable
app.config.update(dict(
    DEBUG=True,
    SECRET_KEY = 'development key',

    PROJECT_FOLDER = os.path.join(os.path.dirname(app.root_path), 'Projects'),
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'data'),
    MAIL_SERVER = MAIL_SERVER,
    MAIL_PORT = MAIL_PORT, 
    MAIL_USE_SSL = MAIL_USE_SSL, 
    MAIL_USERNAME = MAIL_USERNAME, 
    MAIL_PASSWORD = MAIL_PASSWORD,
    MAIL_USE_SSL = True
    ))

mail = Mail(app)

#######################################################################################
# Basic pages
#######################################################################################

@app.route('/')
@app.route('/index')
def index():
    """
    This is the home page, where visitor lands first. 
    """
    return render_template('index.html')


# -------------------------------------------------------------------------------------
@app.route('/glossary')
def glossary():
    """
    This is the glossary page. 
    """
    return render_template('glossary.html')    

    
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
@app.route('/articles_media')
def articles_media():
    """
    This is the media articles page. 
    """
    return render_template('articles_media.html') 


# -------------------------------------------------------------------------------------
@app.route('/videos')
def videos():
    """
    This is the videos page. 
    """
    return render_template('videos.html') 


# -------------------------------------------------------------------------------------
@app.route('/rice_poster')
def rice_poster():
    """
    This is the Rice poster page. Its linked to in articles_media
    """
    return render_template('rice_poster.html')    


# -------------------------------------------------------------------------------------
@app.route('/ledda_framework')
def ledda_framework():
    """
    This is the page for the LEDDA. 
    """
    return render_template('ledda_framework.html') 


# -------------------------------------------------------------------------------------
@app.route('/contact_form', methods=['POST'])
def contact_form():
    """
    This mails the contact form. 
    """
    
    form = dict(request.form)
    
    if not re.match(r"[^@]+@[^@]+\.[^@]+", form['email']):
        test = re.match(r"[^@]+@[^@]+\.[^@]+", form['email'])
        print("\n", form['email'], "    ", test, "\n")
        return jsonify(msg="Email validation fails")
    
    match = re.match(
        "facebook|twitter|instagram|youtube|you tube|design|website|follower|fan|visitor|roi",
        form['message'].lower())

    print("match: ", match)
    if match:
        return jsonify(msg="Message validation fails")
                
    msg = Message("PSP Message", sender= 'info@PrincipledSocietiesProject.org', 
        recipients=['info@PrincipledSocietiesProject.org'])
    msg.body = ("""
        From: {:} <{:}>,
        {:}
        """).format(form['name'], form['email'], form['message'])

    print("test: ", (form['magic'].lower() in ['4', 'four']))
    print((not match))
    print((form['other']==""))
    print(form['other'], form['other'] is None)
    if (form['magic'].lower() in ['4', 'four']) and (not match) and (form['other']==""):    
        mail.send(msg)
        #pass
        print("sent")
        print("msg= ", str(msg))
                     
    else:
        print("msg= ", str(msg))
        return jsonify(msg="Mail not sent. Email validation fails")
    
    return jsonify(msg="OK")


#######################################################################################
# Book EDD
#######################################################################################

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



#######################################################################################
# Misc
#######################################################################################

# -------------------------------------------------------------------------------------
@app.route('/sitemap.xml')
@app.route('/psp_rss.xml')
@app.route('/robots.txt')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


# -------------------------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
        return render_template('404.html'), 404
            



    
#######################################################################################
# Run the steady state model
#######################################################################################

# -------------------------------------------------------------------------------------
@app.route('/model_steady_01a')
def model_steady_state_01a():
    """
    This is the steady_01 page, 1. 
    """
    return render_template('model_steady_01a.html')    


# -------------------------------------------------------------------------------------
@app.route('/model_steady_01b')
def model_steady_state_01b():
    """
    This is the steady_01 page, 2. 
    """
    return render_template('model_steady_01b.html')    


# -------------------------------------------------------------------------------------
@app.route('/model_steady_01c')
def model_steady_state_01c():
    """
    This is the steady_01 page, 3. 
    """
    return render_template('model_steady_01c.html')    


# -------------------------------------------------------------------------------------
@app.route('/runModel', methods=['POST'])
def runModel():
    """
    Run SS model 
    """
    print("jjjjj")
    data = dict(request.form)
    
    K = list(data.keys())
    K.sort()

    # On Apache server, data[key] is a list, not a value. So change here to value.
    for k,v in data.items():
        if isinstance(v, list):
                data[k] = v[0]
        else:
                data[k] = v

    paramsDic = data.copy()
    
    if False:
        # for testing
        print("\ninitial data:")
        for k in K:
            print("k={}, data[k]={}, type={}".format(k, data[k], type(data[k])))
            if (k[0:9] not in ['flexible_']) and (k not in ['doGenetic', 'doBFGS','doRandomStart']):
                    print("     float={}".format(float(data[k])/100.))
    

    # remove lists, make floats, convert % to fractions. Original used paramsDic[k][0]
    _ = [data.__setitem__(k, float(data[k])/100.) \
        for k in K if ((k[0:9] not in ['flexible_']) and (k not in ['doGenetic', 'doBFGS','doRandomStart']))]
    
    # for checkboxes, convert on/off to integers 1/0
    for k in K:
        if (k[0:9] in ['flexible_']) or (k in ['doGenetic', 'doBFGS','doRandomStart']):
            if data[k] == 'true':
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
    

#######################################################################################
# Book CNN
#######################################################################################

# -------------------------------------------------------------------------------------
@app.route('/omp_books')
def omp_books():
    """
    This page is for Oregon Medical Press books by JB. CNN is free to download. This is to avoid
    need for a seperate server for OMP (as it is defunct).
    """
    return render_template('omp_books.html') 


# -------------------------------------------------------------------------------------
@app.route('/omp_book_info')
def omp_book_info():
    """
    Information about OMP books. 
    """
    return render_template('omp_book_info.html')    



'''
# -------------------------------------------------------------------------------------
@app.route('/income_generation')
def income_generation():
    """
    This page is about income distributions are generated for the models. 
    """
    return render_template('income_generation.html') 


# -------------------------------------------------------------------------------------
@app.route('/who_interested')
def who_should_be_interested():
    """
    This is who should be interested page
    """
    return render_template('who_interested.html')    


# -------------------------------------------------------------------------------------
@app.route('/collaborate_engage')
def collaborate_engage():
    """
    This is collaborate_engage page 
    """
    return render_template('collaborate_engage.html')    

# -------------------------------------------------------------------------------------
@app.route('/blog')
def articles_blog():
    """
    This is the blog articles page. 
    """
    return render_template('blog.html') 

# -------------------------------------------------------------------------------------
@app.route('/donation_thanks')
def donation_thanks():
    """
    This is the thanks for donation page. 
    """
    return render_template('donation_thanks.html') 


# -------------------------------------------------------------------------------------
@app.route('/economy_of_meaning')
def economy_of_meaning():
    """
    This is the page for a blog article. 
    """
    return render_template('economy_of_meaning.html') 


# -------------------------------------------------------------------------------------
@app.route('/why_stop_at_basic_income')
def why_stop_at_basic_income():
    """
    This is the page for a blog article. 
    """
    return render_template('why_stop_at_basic_income.html') 
    
        
# -------------------------------------------------------------------------------------
@app.route('/wellbeing_centrality_summary')
def wellbeing_centrality_summary():
    """
    This is the page for a blog article. 
    """
    return render_template('wellbeing_centrality_summary.html') 


# -------------------------------------------------------------------------------------
@app.route('/ideals_democracy_capitalism')
def ideals_democracy_capitalism():
    """
    This is the page for a blog article. 
    """
    return render_template('ideals_democracy_capitalism.html') 
    
    
# -------------------------------------------------------------------------------------
@app.route('/socio_prospectus')
def socio_prospectus():
    """
    This is the page for a blog article. 
    """
    return render_template('socio_prospectus.html') 


# -------------------------------------------------------------------------------------
@app.route('/prospectus')
def prospectus():
    """
    This is the page for the prospectus. 
    """
    return render_template('prospectus.html') 

# -------------------------------------------------------------------------------------
@app.route('/donate')
def donate():
    """
    This is the donate page. 
    """
    return render_template('donate.html') 


'''









     
