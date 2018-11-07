import sys, datetime
from subprocess import Popen,PIPE


def log_details():
  command = ' '.join(sys.argv)
  print(command)

  p = Popen(["/usr/bin/git","log","--pretty=format:\"%H\"","-1"], stdout=PIPE, stderr=PIPE)
  res_string = ""
  res_out,res_err = p.communicate()
  print("{}: git commit: {}".format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), res_out.decode()))

def log_string(s):
  print("{}: git commit: {}".format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), s))
