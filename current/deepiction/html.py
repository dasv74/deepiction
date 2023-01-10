class HTML:

  def __init__(self, filename, title='Training Report'):
    self.f = open(filename, "w")
    self.f.write('<html>\n')
    self.f.write('<head>\n')
    self.f.write(f'<title>{title}</title>\n')
    self.style()
    self.f.write('</head>\n\n')
    self.f.write('<body>\n')

  def close(self):
    self.f.write('</body>\n')
    self.f.write('</html> \n')
    self.f.close()
  
  def p(self, text):
    self.f.write(f'<p>{text}</p>\n')
  
  def h(self, rank, text):
    self.f.write(f'<h{rank}>{text}</h{rank}>\n')
  
  def img(self, image, width=800):
     self.f.write(f'<img src="{image}" style="width:{width}px; border:1px solid #ddd; padding:10px">\n')
     
  def table(self, header):
    self.f.write(f'<table>\n')
    self.f.write(f'<tr>\n')
    for item in header:
      self.f.write(f'<th>{item}</th>')
    self.f.write(f'</tr>\n')
    
  def tr(self, list):
    self.f.write(f'<tr>\n')
    for item in list:
      if isinstance(item, float):
        self.f.write(f'<td>{item:.5g}</td>')
      else:
        self.f.write(f'<td>{item}</td>')
    self.f.write(f'</tr>\n')

  def table_end(self, header):
    self.f.write(f'</table>\n')

  def style(self):
    self.f.write('''<style>
      body {margin:30px; padding: 30px; line-height:1.15; background-color: #fff; color: #212121; font-family: Arial, sans-serif; font-size: 1rem; font-weight: 400; line-height: 1.5; margin: 0; text-align: left}
      b {font-weight:bolder}
      a {background-color: transparent; color:#212121}
      a, a:hover {text-decoration:underline}
      a:hover {color:red}
      a:not([href]):not([class]), a:not([href]):not([class]):hover {color: inherit;text-decoration:none}
      img {border-style:none; vertical-align:middle}
      table {border-collapse:collapse}
      th {text-align: inherit}
      h1, h2, h3, h4, h5, h6 {color: inherit;font-family: inherit;font-weight: 400;line-height: 1.2;margin-top:0;margin-bottom:.5rem}
      h1 {font-size:3.00rem} h2 {font-size:2.33rem} h3 {font-size:1.78rem} h4 {font-size:1.44rem} h5 {font-size:1.17rem} h6 {font-size:1rem}
      table { border-collapse: collapse;margin: 25px 0; font-size: 0.9em; font-family: sans-serif; min-width: 400px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15)}
      thead { background-color: #ADD8E6; color: #212121; text-align: left;}
      tr { background-color: #fff; color: #212121; text-align: left;}
      th, td {padding: 4px 4px;}
      tbody tr {border-bottom: 1px solid #dddddd;}
      </style>''')


