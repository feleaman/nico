# Program opens file mytext.txt and generates mytext.html considering german characters.

#Open txt
mytext = open('mytext.txt', 'r')
texto = mytext.read()

#German and other characters
texto = texto.replace('&', '&amp;')
texto = texto.replace('Ö', '&Ouml;')
texto = texto.replace('ö', '&ouml;')
texto = texto.replace('Ä', '&Auml;')
texto = texto.replace('ä', '&auml;')
texto = texto.replace('ü', '&uuml;')
texto = texto.replace('Ü', '&Uuml;')
texto = texto.replace('ß', '&szlig;')
texto = texto.replace('<', 'lt;')
texto = texto.replace('>', 'gt;')
texto = texto.replace('"', 'quot;')
texto = texto.replace('\n', '<br>')

#Open file Html
f = open('mytext.html','w')

#Generate text for the html file
message = """<html>
<head></head>
<body><p>""" + texto + """</p></body>
</html>"""

#Save and close html
f.write(message)
f.close()

#Additional output
add_text = 'Created by Carla Candia, 2019, FH Aachen'
g = open('readme.txt','w')
g.write(add_text)
g.close()