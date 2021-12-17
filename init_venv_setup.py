from subprocess import run, PIPE, STDOUT

cmd = "pip install -r requirements.txt"
ps = run(cmd, stdout=PIPE, stderr=STDOUT, shell=True, text=True)
print(ps.stdout)

cmd = "python -m spacy download en_core_web_lg"
ps = run(cmd, stdout=PIPE, stderr=STDOUT, shell=True, text=True)
print(ps.stdout)


cmd = "python -m spacy download en_core_web_sm"
ps = run(cmd, stdout=PIPE, stderr=STDOUT, shell=True, text=True)
print(ps.stdout)

