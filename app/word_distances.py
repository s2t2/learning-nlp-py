
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("LOADING SPACY MODEL...")
nlp = spacy.load("en_core_web_md")

def get_word_vectors(words):
    # converts a list of words into their word vectors
    return [nlp(word).vector for word in words]

def show_distances_viz():
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

def term_dependencies_viz(text):
    doc = nlp(text)
    displacy.serve(doc, style="dep")

if __name__ == "__main__":

    #text = """
    #Microsoft releases bumper patches
#
    #Microsoft has warned PC users to update their systems with the latest security fixes for flaws in Windows programs.
#
    #In its monthly security bulletin, it flagged up eight "critical" security holes which could leave PCs open to attack if left unpatched. The number of holes considered "critical" is more than usual. They affect Windows programs, including Internet Explorer (IE), media player and instant messaging. Four other important fixes were also released. These were considered to be less critical, however. If not updated, either automatically or manually, PC users running the programs could be vulnerable to viruses or other malicious attacks designed to exploit the holes. Many of the flaws could be used by virus writers to take over computers remotely, install programs, change, and delete or see data.
#
    #One of the critical patches Microsoft has made available is an important one that fixes some IE flaws. Stephen Toulouse, a Microsoft security manager, said the flaws were known about, and although the firm had not seen any attacks exploiting the flaw, he did not rule them out. Often, when a critical flaw is announced, spates of viruses follow because home users and businesses leave the flaw unpatched. A further patch fixes a hole in Media Player, Windows Messenger and MSN Messenger which an attacker could use to take control of unprotected machines through .png files. Microsoft announces any vulnerabilities in its software every month. The most important ones are those which are classed as "critical". Its latest releases came the week that the company announced it was to buy security software maker Sybari Software as part of Microsoft's plans to make its own security programs.
    #"""

    #text = "all the kings men ate all the kings hens until they all got tired and went to sleep zzz"
    text = "all the kings men ate all the kings hens"
    print(text)

    term_dependencies_viz(text)


    exit()


    show_distances_viz()
