from browser import document, alert, html, ajax, window
import json


##########################################################################################
# Contact form
##########################################################################################

def show_contact(evt):
  """
  Show the contact form
  """
  
  form = html.FORM(id="contact_form", method='post', action='/contact_form')
  
  div = html.DIV()
  label = html.LABEL()
  span = html.SPAN()
  span.text = "Name: (required)"
  widget = html.INPUT(id="name",
    placeholder="Please enter your name", type="text", tabindex="1", required=True, autofocus=True)
  label <= span + widget
  div <= label
  form <= div

  div = html.DIV()
  label = html.LABEL()
  span = html.SPAN()
  span.text = "Email: (required)"
  widget = html.INPUT(id="email",
    placeholder="Please enter your email address", type="email", tabindex="2", required=True)
  label <= span + widget
  div <= label
  form <= div

  div = html.DIV()
  label = html.LABEL()
  span = html.SPAN()
  span.text = "Other: (if you type anything in this box your message will be trashed)"
  widget = html.INPUT(id="other",
    placeholder="Don't do it.", type="text", tabindex="3", required=True)
  label <= span + widget
  div <= label
  form <= div

  div = html.DIV()
  label = html.LABEL()
  span = html.SPAN()
  span.text = "Are you a human? If so, how many legs does a dog have?"
  widget = html.INPUT(id="magic",
    placeholder="How many?", type="text", tabindex="4", required=True)
  label <= span + widget
  div <= label
  form <= div

  div = html.DIV()
  label = html.LABEL()
  span = html.SPAN()
  span.text = "Message"
  widget = html.TEXTAREA(id="message",
    placeholder="Please type your message", tabindex="5", required=True)
  label <= span + widget
  div <= label
  form <= div

  div = html.DIV()
  button = html.BUTTON(type="button", id="contact_submit")
  button.text="Submit Message"
  div <= button
  form <= div
  
  document['contact_container'] <= form
  document['contact_submit'].bind('click', submit_contact)
  window.scrollBy(0,50)


# ---------------------------------------------------------------------------------------
def submit_contact(evt):
  """
  Submit the contact form
  """
  
  document['msg_container'].text = ""
  
  name = document['name'].value.strip()
  email = document['email'].value.strip()
  other = document['other'].value.strip()
  magic = document['magic'].value.strip()
  message = document['message'].value.strip()
    
  if name == "":
    alert("A name is required")
    return
  if email == "":
    alert("An email address is required")
    return
  if message == "":
    alert("A message is required")
    return

  if magic not in ['4', 'four']:
    alert("Really?  How many legs?")
    return
  
  if (email.count('@') != 1) or (email.count('.') == 0) or (email.count(' ')>0):
    alert("Email validation fails (no spaces, must have a '@' and a '.')")
    return  

  data = {
    'name': name,
    'email': email,
    'other': other,
    'magic': magic,
    'message': message}
        
  req = ajax.ajax()
  req.bind('complete', sent_message)
  req.open('POST', '/contact_form', True)
  req.set_header('content-type','application/x-www-form-urlencoded')
  req.send(data)  


# ---------------------------------------------------------------------------------------
def sent_message(req):
  """
  Handle the send message callback
  """
  
  if req.status not in [200, 0]:
    document['msg_container'].html = "<p>An error occurred.</p>"
    document['contact_submit'].disabled = False
    return
  
  data = json.loads(req.text)
  
  if data['msg'] != 'OK':
    document['msg_container'].html = "<p>" + data['msg'] + "</p>"
    document['contact_submit'].disabled = False
    return
  
  document['msg_container'].html = '<p>Thanks, your message has been sent.</>'
  document['contact_submit'].disabled = True
  
  window.scrollBy(0,50)
  
  


##########################################################################################
# Bindings etc.
##########################################################################################
  
document['show_contact'].bind('click', show_contact)


