from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements()->List[str]:
    ## This function will return the list of requirements
    requirement_list:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            ## read lines from requirements.txt
            lines = file.readlines()
            ## process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines ans -e.
                if requirement and not requirement.startswith(HYPHEN_E_DOT):
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print('Requirements.txt not found!!!')
    
    return requirement_list
                    

setup(
    name = 'Bankruptcy Prevention',
    version = '0.0.1',
    packages = find_packages(),
    install_requires = get_requirements()
)