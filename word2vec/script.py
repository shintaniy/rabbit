from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter
from gensim.models import word2vec
import sys


def save_model():
    with open('kokoro.txt', 'rt', encoding='utf-8') as fin:
        content = fin.read()

    token_filters = [POSKeepFilter(['名詞', '代名詞'])]
    a = Analyzer(token_filters=token_filters)

    sentences = [[tok.surface for tok in a.analyze(content)]]
    sentences[0].extend(['心配'])

    model = word2vec.Word2Vec(sentences,  min_count=5, window=5 )
    model.save('kokoro.model')
    print('saved at "kokoro.model"')


def show_most_similar():
    model = word2vec.Word2Vec.load('kokoro.model')
    results = model.wv.most_similar(negative=['私', '心配'])
    for result in results:
        print(result)


def usage():
    print('''Sample of Word2Vec

Usage:

    script.py [command]

The commands are:

    save-model
    most-similar''')


def main():
    if len(sys.argv) < 2:
        return usage()

    cmd = sys.argv[1]
    if cmd == 'save-model':
        save_model()
    elif cmd == 'most-similar':
        show_most_similar()


main()