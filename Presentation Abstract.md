# Presentation Abstract for  “Are humans good at regulations grouping?”

## AI Abstract

When regulatory frameworks grow beyond the scale manageable by individuals, effective grouping becomes critical. Well-structured groups improve clarity, facilitate assignment of responsibility, and ensure consistency. But how well do human-assigned groupings reflect the actual relationships and structures within a framework?

This presentation tackles this question using an automated approach combining NLP and Graph Analysis. We leverage NLP to automatically identify intricate connections between regulations, moving beyond explicitly stated relationships. These connections form a network graph, which we analyze using community detection methods to uncover data-driven groupings.

By comparing these algorithmically derived groups against existing human-created structures in a case study (using NIS800-53 as an example), we evaluate their effectiveness. The results indicate potential limitations in human-scale grouping and demonstrate how NLP and Graph Analysis can reveal alternative, potentially more coherent ways to organize regulations. This highlights the value of data-driven methods for optimizing regulatory framework structure.

## Lior Abstract

Are humans good at regulations grouping? Can we use a combination of graph analysis and NLP methods to see the group's structure within a regulation framework and how it compares to the human-made structure? When we have a lot of regulations, so no single individual can be responsible for all of them, we will want to split the regulations into well-connected groups so the regulations inside each group will be as similar as possible and will also be different from other groups.

Using NLP methods, we infer the connections between regulations within a framework faster and more thoroughly than humanly possible. And using graph analysis, we turn the connections into a relations graph and discover the best regulations groups within it.
We also compare the results to the human-made ones to see which groups are better, and what the scores and insights we get from the human-made route are compared to our new human-independent route.

## Who am I?

I am lior, and this project came from a problem we had related to a product I work on.

### Disclaimer

Due to the project's results, I was asked to say it is not related to the company in the unit I work for!

## Back story

In 2023, as part of the product, we made NLP embeddings of the cyber regulations in the product. To the quality of the embeddings, we tried to use them for a grouping task and try to “rediscover” the groups we had in the data, but we ran into trouble because, regardless of what we tried, the groups we got from the embedding were totally different from the ones that are stated in the data. After that, this part was abundant.
But in the middle of 2024, I decided to revisit the project with all of the latest tools and knowledge.

## Intro

The project works with the cyber regulation of NIS800-53.
The cleaned version contains 1007 rows of regulation data that have 3 levels:

* 1007 indicators
* 298 controls
* 20 families

A good thing about the NIS800-53 regulation is that the data contains a column of Related Controls that gives us a “golden standard” of all the controls each indicator is said to be related to.

## Why use graph analysis over normal n-dimensional clustering?

* For some cases/methods it is easier to obtain closely related entities (which are needed for graph analysis) than to obtain a good n-dimensional representation of the entities (which is needed for normal n-dimensional clustering).
* Graph-based methods make it easier to combine multiple different inputs into the clustering analysis (making it easier to combine any number of related controls discovery methods we want).
* Graph space can be a non-metric space, and we see it as an advantage.
* For our chose example we have the exsect data needed for it 🙂.

## Graph analysis of the “golden standard”

Because the regulation framework we chose contains a “golden standard” of the relations between regulations, we will use it in a graph analysis pipeline and see if we can recreate the given families in the regulation framework using the “golden standard” relations.

### “Golden Standard” relations within/between the given families

We want to explore the “golden standard” relations within and between the given families. How many of the “golden standard” relations are within families compared to between families? And how many of them are mutual / non-mutual relations?

(spoiler, bad results)

### How does the relations graph look?

We first build the relations graph.

(spoiler, a mass with non-connected controls)

### Run communities discovery algorithms on the relations graph

After building a relations graph based on the “golden standard” relations, we want to see if we can use communities discovery algorithms to rediscover the families that are given to us in the regulation framework.

(spoiler, all the methods tried failed in the rediscovery task)

### Mid Conclusions

* The “golden standard” relations and the “golden standard” families don't seem to connect
* Either the “golden standard” relations and/or he “golden standard” families have problems.

## Now, using NLP methods, infer the connections between regulations

If the “golden standard” relations are bad and don’t work, let us make our own relations using NLP methods.

We will also make a NLP pipeline to automate the process.

### The three methods used

* BM25 - A representation of classical NLP methods that will be best for rare words/phrases similarly.
* HuggingFace🤗 Sentence Transformers - A representation of small open-source NLP embedding model, taken from the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
* SaaS NLP embedding service - A representation of a SaaS NLP embedding service that can use a match bigger model than what we can run locally.

### Make a better relations graph

Now we can make a new better relations graph that can take into account all of our previous relations sources.

We say that using multiple relations sources is better than single source because it can help us with the single source bias.

### Run communities discovery algorithms on the better relations graph

Now we will run the communities discovery we did least time but on our new relations graph to try and rediscover the families that are given to us in the regulation framework.

(spoiler, we still can not rediscover the "correct families")

### Conclusions After NLP pipeline

* The NLP pipeline automation helps to do things faster with less effort.
* The relations graph is still a mess, but slightly better for the original “golden standard” families.

## Final Conclusions

* Humans are not so good at regulations grouping
* The NLP + Graph Analysis can find better groupings then the human ones
* there are indications that 1 to 1 groupings may not work best at the case of regulations grouping
