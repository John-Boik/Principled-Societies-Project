from browser import document, alert, confirm, prompt
from browser import window
import json

# -------------------------- menu -------------------------------------------
def menu():
  current = window.location.pathname.split("/")[1]
  if current == "":
    # use manage page as default index
    current = "index"

  elem = document["menu_" + current]
  elem.style.backgroundColor = 'DarkOrange '
  
  currentColor = '#03964b'
  
  if current in ['about', 'index', 'contact']:
    document['menu_' + 'index'].style.backgroundColor = currentColor

  if current in ['steady_01']:
    document['menu_' + 'interactive'].style.backgroundColor = currentColor  
  
  if current in ['engage', 'contribute', 'volunteer', 'collaborate', 'jobs', 'invite']:
    document['menu_' + 'involved'].style.backgroundColor = currentColor
  

# ==================== bindings =======================================  

menu()





