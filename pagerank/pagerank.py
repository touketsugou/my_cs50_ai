import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    result = dict()
    N = len(corpus)
    
    # 处理无出链的情况
    if not corpus[page]:
        return {p: 1/N for p in corpus}
        
    # 基础概率
    for p in corpus:
        result[p] = (1-damping_factor) / N
    
    # 链接概率
    for p in corpus[page]:
        result[p] += damping_factor / len(corpus[page])
    
    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    count_dic = dict()
    for p in corpus:
        count_dic[p] = 0

    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        pages_distribution = transition_model(corpus=corpus, page=current_page, damping_factor=damping_factor)
        pages = list(pages_distribution.keys())
        weights = list(pages_distribution.values())
        current_page = random.choices(pages, weights=weights, k=1)[0]
        count_dic[current_page] = count_dic[current_page] + 1
    
    result = dict()

    for p in count_dic:
        result[p] = count_dic[p] / n

    return result


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = dict()

    for page in corpus:
        page_rank[page] = 1 / len(corpus)
    
    new_page_rank = iterate_transition(corpus=corpus, page_rank=page_rank, damping_factor=damping_factor)
    while(not(should_stop(page_rank, new_page_rank))):
        page_rank = new_page_rank
        new_page_rank = new_page_rank = iterate_transition(corpus=corpus, page_rank=page_rank, damping_factor=damping_factor)

    return new_page_rank

def iterate_transition(corpus, page_rank, damping_factor):
    page_sum = len(page_rank)
    new_page_rank = dict()
    for p in page_rank:
        new_page_rank[p] = (1-damping_factor)/page_sum
        lps = link_pages(corpus, p)
        for link_page in lps:
            new_page_rank[p] = new_page_rank[p] + damping_factor*page_rank[link_page]/len(corpus[link_page])

    return new_page_rank

def should_stop(previous_page_rank, page_rank):
    sum = 0
    for page in page_rank:
        if abs(previous_page_rank[page]-page_rank[page]) < 0.0001:
            sum = sum + 1
    
    return sum == len(page_rank)

def link_pages(corpus, page):
    result = []
    for key in corpus:
        if page in corpus[key]:
            result.append(key)
    return result


if __name__ == "__main__":
    main()
