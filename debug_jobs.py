import ast

with open('data/summaries&archives/jobs.txt', 'r') as f:
    content = f.read().strip()
    jobs = ast.literal_eval(content)

