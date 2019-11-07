# https://github.com/LambdaSchool/DS-Unit-4-Sprint-1-NLP/tree/master/module4-topic-modeling
# https://learn.lambdaschool.com/ds/module/recbYIWnPYs2J4AWC/
#
# Topic Modeling
#
# At the end of this module, you should be able to:
#   + describe the latent dirichlet allocation process
#   + implement a topic model using the gensim library
#   + interpret document topic distributions and summarize findings

# Part 1: Describe how an LDA Model works
# Part 2: Estimate a LDA Model with Gensim
# Part 3: Interpret LDA results
# Part 4: Select the appropriate number of topics

import os

NOVELS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data", "novels")

if __name__ == "__main__":
    print("TOPIC MODELING, YO")
