from subprocess import run, PIPE, STDOUT

ps = run("pip install -r requirements.txt",
         stdout=PIPE, stderr=STDOUT, shell=True, text=True)
print(ps.stdout)

ps = run("python -m spacy download en_core_web_lg",
         stdout=PIPE, stderr=STDOUT, shell=True, text=True)
print(ps.stdout)


ps = run("python -m spacy download en_core_web_sm",
         stdout=PIPE, stderr=STDOUT, shell=True, text=True)
print(ps.stdout)

