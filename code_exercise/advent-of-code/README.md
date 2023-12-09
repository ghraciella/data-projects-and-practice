# adventofcode 

Update the readme to your needs

* template repository from [Pyladies Hamburg]()


## How it works (2023)

Trying out to practice testing more so.. here it's going to be all test first!

For each day run:

```bash
# testing day 1 on test data (from site)
pytest 2023/01.py

# running day 1 on day 1 data

python 2023/py
```

## Setup

First time.. setting things up with black and flake8 for formatting and pre-commit for fixing things before they getting committed to git.

Requirements:

- python 3.12.0

```bash

# (optional) to avoid error 'python-build: definition not found: 3.12.0'
brew update && brew upgrade pyenv

# to avoid make error 
# install required python version
pyenv install 3.12.0

# to setup venv and install stuff
make setup 

```


Setting up pre-commit (it uses the .pre-commit-config.yaml which specify rules for black and flake8 and flake8 uses .flake8 which has line length rules set so it doesn't clash with black)

```bash
pre-commit
```

## Getting the AOCD work

After logging in to adventofcode on firefox or chrome get the token, see [how to](https://github.com/wimglenn/advent-of-code-wim/issues/1) and add the export to your profile (you need to update when token expires.. after a year basically)


You can run this in your terminal
````bash
export AOC_SESSION=cafef00db01dfaceba5eba11deadbeef```

````

## How to use this

For each day copy the 00.py, rename it and change that it gets data for that day, and the tests expect the value in the challenges.

If your code works then running

```pytest 2023```

should return all tests passed.