"""
Clean up functions
The translation generated some multiple lines if it was stopped and started again
"""


def get_index(in_str, i):
    ret = -1
    try:
        x = in_str.split(";")[0]
        ret = int(x)
    except ValueError:
        print("[Error]", i, in_str)
    return ret


def clean_doublelines(ds):
    del_count = 0
    len_ds = len(ds)
    for i in range(1,len_ds):
        if get_index(ds[i-1],i-1) ==  get_index(ds[i],i):
            print("del: {}".format(i))
            del(ds[i])
            del_count += 1
        if i+1 == len_ds - del_count:
            print("Last i ", i)
            break
    return ds


def clean_double_lines_sueddeutsche_data(name, language):
    article_path = "../data/sueddeutsche_old/articles_{}_{}".format(language, name)
    highlights_path = "../data/sueddeutsche_old/highlights_{}_{}".format(language, name)

    articles = [x.rstrip() for x in open(article_path).readlines()]
    highlights = [x.rstrip() for x in open(highlights_path).readlines()]
    articles = clean_doublelines(articles)
    highlights = clean_doublelines(highlights)
    with open("../data/sueddeutsche_old/articles_{}_{}_cleaned".format(language, name), "a") as file:
        for line in articles:
            file.write(line + "\n")

    with open("../data/sueddeutsche_old/highlights_{}_{}_cleaned".format(language, name), "a") as file:
        for line in highlights:
            file.write(line + "\n")

    return articles, highlights

if __name__ == '__main__':
    train_articles, train_highlights = clean_double_lines_sueddeutsche_data("train", "en")
    test_articles, test_highlights = clean_double_lines_sueddeutsche_data("test", "en")
    val_articles, val_highlights = clean_double_lines_sueddeutsche_data("val", "en")

