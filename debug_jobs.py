import ast

with open('data/jobs.txt', 'r') as f:
    content = f.read().strip()
    jobs = ast.literal_eval(content)

