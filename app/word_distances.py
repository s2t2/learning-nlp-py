



from sklearn.decomposition import PCA
import spacy
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_md")

def get_word_vectors(words):
    # converts a list of words into their word vectors
    return [nlp(word).vector for word in words]

if __name__ == "__main__":

    words = ['car', 'truck', 'suv', 'race', 'elves', 'dragon', 'sword', 'king', 'queen', 'prince', 'horse', 'fish' , 'lion', 'tiger', 'lynx', 'potato']
    vectors = get_word_vectors(words)

    # init pca model and tell it to project data down onto 2 dimensions
    pca = PCA(n_components=2)

    # fit the pca model to our 300D data, this will work out which is the best
    # way to project the data down that will best maintain the relative distances
    # between data points. It will store these instructions on how to transform the data.
    pca.fit(vectors)

    # Tell our (fitted) pca model to transform our 300D data down onto 2D using the
    # instructions it learnt during the fit phase.
    word_vecs_2d = pca.transform(vectors)

    # let's look at our new 2D word vectors
    print(word_vecs_2d)

    #
    # DATAVIZ
    #

    #plt.figure(figsize=(20,15))
    plt.figure()

    # plot the scatter plot of where the words will be
    plt.scatter(word_vecs_2d[:,0], word_vecs_2d[:,1])

    # for each word and coordinate pair: draw the text on the plot
    for word, coord in zip(words, word_vecs_2d):
        x, y = coord
        plt.text(x, y, word, size= 15)

    # show the plot
    plt.show()




    #doc1 = nlp("It's a warm summer day")
    #doc2 = nlp("It's sunny outside")
    ## Get the similarity of doc1 and doc2
    #similarity = doc1.__
    #print(similarity)
